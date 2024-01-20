import torch
import torch.nn as nn
from utils import intersection_over_union as iou


class Yolov1Loss(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20, lambda_coord=5.0, lambda_noobj=0.5):
        super(Yolov1Loss, self).__init__()
        self.S, self.B, self.C = split_size, num_boxes, num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, predicts, labels):

        iou_box1 = iou(predicts[..., 21:25], labels[..., 21:25])
        iou_box2 = iou(predicts[..., 26:30], labels[..., 21:25])
        iou_boxes = torch.cat([iou_box1.unsqueeze(0), iou_box2.unsqueeze(0)], dim=0)
        # highest_iou, idx : torch.Size([batch_size, 7, 7, 1])
        highest_iou, highest_iou_idx = torch.max(iou_boxes, dim=0)
        label_object_exist = labels[..., 20:21]

        predict_box = label_object_exist * (
                highest_iou_idx * predicts[..., 26:30]
                + (1 - highest_iou_idx) * predicts[..., 21:25]
        )
        label_box = label_object_exist * labels[..., 21:25]

        # x, y, w^0.5, h^0.5
        predict_box[..., 2:4] = (torch.sign(predict_box[..., 2:4])
                                 * (torch.sqrt(torch.abs(predict_box[..., 2:4] + 1e-6))))
        label_box[..., 2:4] = (torch.sqrt(label_box[..., 2:4]))

        # x,y,w,h box loss
        box_loss = self.mse(
            torch.flatten(predict_box, end_dim=-2),
            torch.flatten(label_box, end_dim=-2)
        )

        # probability loss
        predict_prob = (highest_iou_idx * predicts[..., 25:26]
                        + (1 - highest_iou_idx) * predicts[..., 20:21])
        obj_loss = self.mse(
            torch.flatten(label_object_exist * predict_prob),
            torch.flatten(label_object_exist * labels[..., 20:21])
        )

        # probability miss penalty
        no_obj_loss = self.mse(
            torch.flatten((1 - label_object_exist) * predicts[..., 20:21], end_dim=-2),
            torch.flatten((1 - label_object_exist) * labels[..., 20:21], end_dim=-2)
        )
        no_obj_loss = no_obj_loss + self.mse(
            torch.flatten((1 - label_object_exist) * predicts[..., 25:26], end_dim=-2),
            torch.flatten((1 - label_object_exist) * labels[..., 20:21], end_dim=-2)
        )

        # class loss
        class_loss = self.mse(
            torch.flatten(label_object_exist * predicts[..., 0:20], end_dim=-2),
            torch.flatten(label_object_exist * labels[..., 0:20], end_dim=-2)
        )

        loss = (self.lambda_coord * box_loss + obj_loss + self.lambda_noobj * no_obj_loss + class_loss)

        return loss

