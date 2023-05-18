# Foreground Image Segmentation with UNet Architecture and MobileNetV2 Weights

This is an old TensorFlow project from 2020 using a modified [UNet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) autoencoder architecture with MobileNetV2 weights to perform transfer learning for foreground image segmentation.
New and improved techniques and models, like Meta's [Segment Anything Model (SAM)](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/) have been rolled out recently, so this approach is not SOTA, but it makes for a good learning experience.

The original usecase was polyp segmentation from colonoscopy images (_I know, not fun_ 😅), but this was repurposed for foreground segmentation of everyday objects.

The 2 main resources that helped me build this were:
1. The official [Tensorflow Segmentation Turtorial](https://www.tensorflow.org/tutorials/images/segmentation)\*
2. Idiot Developer's [Polyp Segmentation using UNET tutorial](https://idiotdeveloper.com/polyp-segmentation-using-unet-in-tensorflow-2/)\*

---
Example inputs and outputs:
<table>
  <tr>
    <th>Input</th>
    <th>Target Mask</th>
    <th>Output</th>
  </tr>
  <tr>
    <td><img src="https://github.com/KayO-GH/foreground-image-segmentation/assets/18174012/a49c583c-2dad-4989-84f1-a4c3f0c95fd4"/></td>
    <td><img src="https://github.com/KayO-GH/foreground-image-segmentation/assets/18174012/63501fcf-493c-4b85-9967-579cd45fd35b"></td>
    <td><img src="https://github.com/KayO-GH/foreground-image-segmentation/assets/18174012/35a5209a-bf59-459a-89d2-66e75a0fc254"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/KayO-GH/foreground-image-segmentation/assets/18174012/6e37da25-2728-40df-ac3f-3b494d8b72e0"></td>
    <td><img src="https://github.com/KayO-GH/foreground-image-segmentation/assets/18174012/b01015dc-be4e-41d1-8994-34f740cbd497"></td>
    <td><img src="https://github.com/KayO-GH/foreground-image-segmentation/assets/18174012/9bd77f93-2892-4b37-811e-2d24d84a2ad2"/></td>
  </tr>
</table>

---

**\*Disclaimer:** _It's been almost 3 years since I worked on this actively, and there is a chance that both resources referenced above have changed (read "improved") since then._

_Extra note: Upon visual inspection  it may appear that the images in `train_images` are the same as those in `test/input_image`. The former contains 74 images and is a subset of the latter which contains 334 images. The task at hand required us to have image resources allocated this way for training and testing, but this is not usually how things are done in practice._

_Shoutout to Wuyeh Jobe, who helped me understand some pieces of the code I found confusing at the time._
