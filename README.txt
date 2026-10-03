CNN × VIT — MODEL OBSERVATORY

Open cnn-vit.html in a browser. This self-contained file works offline.

WHAT YOU CAN EXPLORE
- Choose one of the seven project images, or upload one or more images.
- Both trained models process the same resized image, with each model's own normalization.
- Build both models animates the CNN blocks and the ViT patches/tokens.
- Drag either 3D scene to rotate; use the stage selector to isolate a layer and the zoom slider to inspect it.
- Compare all seven class probabilities side by side.
- Inspect every CNN channel in the feature map gallery.
- Select a ViT encoder block, head, and query token to see its actual attention distribution.
- View the 65x65 attention matrix and the 65x96 token embedding matrix.
- Compare parameter counts and evaluation figures saved in your project.

DATA AND INTERPRETATION
CNN: 361,095 trainable parameters; four convolution blocks.
ViT: 273,127 trainable parameters; 64 image patches plus one class token; 96 embedding dimensions; three pre-norm transformer blocks with four attention heads.

CNN input uses RGB values 0–1. ViT input is normalized to −1–1. Uploaded image resizing uses the browser and can differ slightly from Pillow. Uploads stay local and are not retained after reload. The HTML embeds both trained checkpoints, so sharing it also shares their weights.

ViT attention is read directly from the softmax attention matrices, before attention dropout (which is inactive during evaluation). The overlay excludes the class-token column; its attention mass is shown separately. Attention is not a causal explanation or a confidence score. Heatmaps use relative per-view scales. Token colors in 3D show RMS activation magnitude, not predicted class.

The CNN 3D scene samples up to six evenly spaced channels and up to 12x12 cells per map. The gallery retains all channels and original spatial resolution. The ViT scene shows every patch and token; patch textures are reduced for display. Both animations illustrate progression, not execution timing. Dropout is inactive in evaluation.

Saved accuracy and F1 figures are read from results/cnn_metrics.json and results/vit_metrics.json. No new benchmark was run. Both files report 584 test images; identical membership was not independently verified. Parameter counts are calculated from the loaded models. Current checkpoint SHA-256 hashes are embedded in the HTML data.

REBUILD
Use the project's Python environment:

"/Users/daryn/Desktop/midterm_project 2/.venv/bin/python" build_comparison.py --project "/Users/daryn/Desktop/midterm_project 2"

Keep build_comparison.py, comparison-template.html, inference.js, and viewer.js together. The project folder must contain cnn_model.py, vit_model.py, models/registry.json, model checkpoints, test_images, and results metrics. Rebuild after changing trained weights.

VALIDATION
Both checkpoints load strictly against their model definitions. All seven bundled examples have captured results verified against standard PyTorch forward passes. On an identical numerical input, independent JavaScript calculations matched PyTorch at all four ViT token states, all three attention matrices, and the final logits (maximum absolute error below 0.000011); every attention row sums to one. CNN browser probabilities also matched the current checkpoint. Generated JavaScript passes syntax checks. Browser visual verification was unavailable under the session's local-file restrictions.
