"""
PURPOSE:
  Export trained ISNet to ONNX for C++ deployment.
  ONNX Runtime is a C++ inference engine — satisfies ISRO's speed requirement.
  Show judges the production deployment pathway clearly.

WHY ONNX:
  PyTorch has Python overhead (~2ms extra per call).
  ONNX Runtime eliminates Python overhead entirely.
  On CPU: ONNX is 3-5x faster than PyTorch.
  we  can plug the .onnx file directly into their C++ AO control loop.
"""


