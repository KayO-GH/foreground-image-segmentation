# Foreground Image Segmentation with UNet Architecture and MobileNetV2 Weights

This is an old TensorFlow project from 2020 using a modified [UNet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) autoencoder architecture with MobileNetV2 weights to perform transfer learning for foreground image segmentation.
New and improved techniques and models, like Meta's [Segment Anything Model (SAM)](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/) have been rolled out recently, so this approach is not SOTA, but it makes for a good learning experience.

The original usecase was polyp segmentation from colonoscopy images (_I know, not fun_ 😅), but this was repurposed for foreground segmentation of everyday objects.

The 2 main resources that helped me build this were:
1. The official [Tensorflow Segmentation Turtorial](https://www.tensorflow.org/tutorials/images/segmentation)
2. Idiot Developer's [Polyp Segmentation using UNET tutorial](https://idiotdeveloper.com/polyp-segmentation-using-unet-in-tensorflow-2/)

**Note:** _It's been almost 3 years since I did this, and there is a chance that both resources referenced above have changed (read "improved") since then._

_Shoutout to Wuyeh Jobe, who helped me understand some pieces of the code I found confusing at the time._