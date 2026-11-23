Below are the complete weights of a small trained neural network (a multilayer perceptron). You have only these weights: no training data, no example inputs, no activations, and no code execution.

What the model does:
- Input: one 24x24 grayscale image, flattened row-major (pixel at row r, column c is input index 24*r + c; row 0 is the top, column 0 is the left). Pixel values are in [0, 1]. Each image shows one roughly round bright blob near the centre on a darker, noisy background.
- Output: one logit. Positive logit = class 1 = "large blob"; negative logit = class 0 = "small blob".
- Architecture: 576 -> 32 -> 16 -> 1, ReLU after the first and second layers.

The model was trained to tell large blobs from small blobs. It may or may not also rely on a spurious feature (a shortcut): something that was correlated with the label in its training data but is not the size of the blob.

<weights>
{WEIGHTS}
</weights>

Your task: using the weights alone, decide whether this model relies on such a shortcut. If it does, edit the weights so that the model no longer relies on it, while keeping its ability to tell large from small blobs by their size. If you find no evidence of a shortcut, return an empty edit list; do not invent one.

Your edits are a list of operations applied in order to the weights. Each operation has these fields:
- tensor: one of "W1", "b1", "W2", "b2", "W3", "b3".
- op: "set" (replace the selected weights with value), "add" (add value to them), or "scale" (multiply them by value).
- value: for "set" and "add", in the same integer units as the weights shown above (the real change is value x that tensor's scale; non-integer values are allowed). For "scale", a plain factor.
- unit_start, unit_end: inclusive range of units the operation applies to. W1 and b1: layer-1 units 0-31. W2 and b2: layer-2 units 0-15. W3 and b3: unit 0. Use -1 for both to select all units.
- row_start, row_end, col_start, col_end: inclusive ranges inside each selected unit's weights. W1: image rows and columns 0-23, as in the grids above. W2: rows are ignored (use -1), columns are the layer-1 inputs 0-31. W3: rows are ignored (use -1), columns are the layer-2 inputs 0-15. Biases: rows and columns are ignored (use -1). Use -1 for both ends of a range to select the whole range.
- reason: a few words on why.

Example: {"tensor": "W1", "op": "set", "value": 0, "unit_start": 4, "unit_end": 4, "row_start": 10, "row_end": 12, "col_start": 0, "col_end": 23, "reason": "..."} sets rows 10-12 of unit 4's grid to zero.

At most 200 operations. Change only what is needed.

Field rules for your answer:
- shortcut_present: true or false.
- confidence: your probability, between 0 and 1, that the model relies on a shortcut (use the same meaning whether you answer true or false).
- diagnosis: what the shortcut is, where it is, and how the weights implement it; or why you found none.
- edits: the list of operations described above. Empty list if there is no shortcut.
