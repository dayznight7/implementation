import torch


def intersection_over_union(predict, label, box_format="midpoint"):

    """

    :param predict: torch.Size([-1, 4])
                    4 : (x, y, w, h)
    :param label:   torch.Size([-1, 4])
    :return:        torch.Size([-1, 1])
                    1 : (intersection / union)
    """

    if box_format == "midpoint":
        predict_xmin = predict[..., 0:1] - predict[..., 2:3] / 2
        predict_ymin = predict[..., 1:2] - predict[..., 3:4] / 2
        predict_xmax = predict[..., 0:1] + predict[..., 2:3] / 2
        predict_ymax = predict[..., 1:2] + predict[..., 3:4] / 2
        label_xmin = label[..., 0:1] - label[..., 2:3] / 2
        label_ymin = label[..., 1:2] - label[..., 3:4] / 2
        label_xmax = label[..., 0:1] + label[..., 2:3] / 2
        label_ymax = label[..., 1:2] + label[..., 3:4] / 2

    if box_format == "corners":
        predict_xmin = predict[..., 0:1]
        predict_ymin = predict[..., 1:2]
        predict_xmax = predict[..., 2:3]
        predict_ymax = predict[..., 3:4]
        label_xmin = label[..., 0:1]
        label_ymin = label[..., 1:2]
        label_xmax = label[..., 2:3]
        label_ymax = label[..., 3:4]

    overlap_xmin = torch.max(predict_xmin, label_xmin)
    overlap_ymin = torch.max(predict_ymin, label_ymin)
    overlap_xmax = torch.min(predict_xmax, label_xmax)
    overlap_ymax = torch.min(predict_ymax, label_ymax)

    intersection = (overlap_xmax - overlap_xmin).clamp(min=0) * (overlap_ymax - overlap_ymin).clamp(min=0)
    predict_area = abs((predict_xmax - predict_xmin) * (predict_ymax - predict_ymin))
    label_area = abs((label_xmax - label_xmin) * (label_ymax - label_ymin))

    return intersection / (predict_area + label_area - intersection + 1e-6)


def calculate_tp_fp_fn_loss(loader, model, criterion, iou_threshold=0.4, prob_threshold=0.5, device="cuda"):

    model.eval()
    TP, FP, FN = 0, 0, 0
    LOSS = 0.0

    for idx, (inputs, labels) in enumerate(loader):

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        tmp = torch.cat((outputs[..., :20], (outputs[..., 20:25] + outputs[..., 25:30]) / 2.0), dim=-1)
        tmp = torch.cat((tmp, intersection_over_union(tmp[..., 21:25], labels[..., 21:25])), dim=-1)

        true_p = torch.sum(
            torch.logical_and(tmp[..., 20] >= prob_threshold, tmp[..., 25] >= iou_threshold)
        ).item()
        TP += true_p

        false_p = torch.sum(outputs[..., 20] >= prob_threshold).item() - true_p
        FP += false_p

        false_n = torch.sum(labels[..., 20] == 1).item() - true_p
        FN += false_n

        loss = criterion(outputs, labels)
        LOSS += loss.item()

    return TP, FP, FN, LOSS

