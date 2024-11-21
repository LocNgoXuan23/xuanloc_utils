import cv2
import numpy as np
from typing import Union

from .color import Color, ColorPalette
from .common import poly2box

# ///////////////////////////////////////////
class BoxAnnotator:
    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 4,
        text_color: Color = Color.black(),
        text_scale: float = 1,
        text_thickness: int = 2,
        text_padding: int = 10,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.font = cv2.FONT_HERSHEY_SIMPLEX


    def annotate(self, img=None, box=None, mask=None, text=None, c=None):
        if box:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        else:
            x1, y1, x2, y2 = poly2box(mask)
        
        color = (
            self.color.by_idx(c)
            if isinstance(self.color, ColorPalette)
            else self.color
        )

        if box is not None:
            cv2.rectangle(
                img=img,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )

        if mask is not None:
            img = self.draw_polygon(img, mask, color)

        if text is not None:
            lines = text.split('\n')
            line_heights = []
            max_width = 0
            total_height = self.text_padding * (len(lines) + 1)

            for line in lines:
                (text_width, text_height), _ = cv2.getTextSize(
                    text=line,
                    fontFace=self.font,
                    fontScale=self.text_scale,
                    thickness=self.text_thickness,
                )
                max_width = max(max_width, text_width)
                total_height += text_height
                line_heights.append(text_height)

            # Draw background rectangle
            cv2.rectangle(
                img=img,
                pt1=(x1, y1 - total_height),
                pt2=(x1 + max_width + 2 * self.text_padding, y1),
                color=color.as_bgr(),
                thickness=cv2.FILLED,
            )

            # Draw text
            y = y1 - total_height + self.text_padding
            for line, height in zip(lines, line_heights):
                cv2.putText(
                    img=img,
                    text=line,
                    org=(x1 + self.text_padding, y + height),
                    fontFace=self.font,
                    fontScale=self.text_scale,
                    color=self.text_color.as_rgb(),
                    thickness=self.text_thickness,
                    lineType=cv2.LINE_AA,
                )
                y += height + self.text_padding

        return img
    
    def draw_polygon(self, img, points, color):
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        img = cv2.polylines(img, [points], True, color.as_bgr(), self.thickness)
        return img
