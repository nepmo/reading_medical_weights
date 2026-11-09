Below are the complete weights of a small trained neural network (a multilayer perceptron). You have only these weights: no training data, no example inputs, no activations, and no code execution.

What the model does:
- Input: one 24x24 grayscale image, flattened row-major (pixel at row r, column c is input index 24*r + c; row 0 is the top, column 0 is the left). Pixel values are in [0, 1]. Each image shows one roughly round bright blob near the centre on a darker, noisy background.
- Output: one logit. Positive logit = class 1 = "large blob"; negative logit = class 0 = "small blob".
- Architecture: 576 -> 32 -> 16 -> 1, ReLU after the first and second layers.

The model was trained to tell large blobs from small blobs. It may or may not also rely on a spurious feature (a shortcut): something that was correlated with the label in its training data but is not the size of the blob.

<weights>
{WEIGHTS}
</weights>

Your task: using the weights alone, decide whether this model relies on such a shortcut.

If it does, report where in the image the shortcut is, what kind of feature it is, which class it pushes the output toward, and how the weights implement it. If you find no evidence of a shortcut, say so; do not invent one.

Field rules for your answer:
- shortcut_present: true or false.
- confidence: your probability, between 0 and 1, that the model relies on a shortcut (use the same meaning whether you answer true or false).
- location_description: where in the image the shortcut is, in words. Empty string if none.
- row_start, row_end, col_start, col_end: inclusive pixel bounding box (0 to 23) of the shortcut region. Use -1 for all four if there is no shortcut.
- shortcut_type: what kind of feature the shortcut is, in a few words. Empty string if none.
- direction: "increases_class_1" if the shortcut pushes the output toward class 1 (large blob), "increases_class_0" if toward class 0 (small blob), "none" if there is no shortcut, "unclear" otherwise.
- mechanism: how the weights implement the shortcut (which units, which pathway). Empty string if none.
- evidence: the specific weight patterns that support your answer.
- main_feature: in one or two sentences, what the model uses to tell large from small blobs.
