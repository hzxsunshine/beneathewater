# beneathewater

## Prompts
- White balance: The model is trained to accurately represent colors, avoiding blue/green color casts. `["Accurate Color Representation", "Blue/Green Color Cast"]`
- Vibrant and vivid: The model is trained to make the images vibrant and vivid, avoiding dull and washed-out colors. `["Vibrant and Vivid", "Dull and Washed-out"]` and `["Bright, Colorful Underwater Scene", "Dim, Desaturated Underwater Colors"]`
- Exposure and lit: The model is trained to make the scene crystal-clear and unobstructed, avoiding murky and obscured views. It also aims to balance and well-lit the view, avoiding underlit scenes and backlit objects.  `["Crystal-clear and Unobstructed", "Murky and Obscured"]` and `["Balanced and Well-Lit view", "Underlit scene and Backlit Objects"]`
- Sharpness: The model is trained to make the details in the image richly detailed, avoiding blurry and indistinct details. It also aims to make aquatic details sharp, avoiding blurred movement. `["Richly detailed", " Blurry and indistinct"]` and `["Sharp Aquatic Details", "Blurred Movement"]`
- Lighting: The model is trained to make the lighting in the image natural and ambient, avoiding artificial lighting. `["Natural and Ambient Lighting", "Artificial Lighting"]`

## Results
- Base model is the Pretrained [WaterNet](https://github.com/tnwei/waternet) was trained on the [UIEB Dataset](https://li-chongyi.github.io/proj_benchmark.html) for 400 epochs.

- Color-Enhanced model is the Base model with the contrastive loss of colorfulness contrastive pairs.

- WhiteBalance-Enhanced model is the Base model with the contrastive loss of white balance contrastive pairs.

- Exposure-Enhanced model is the Base model with the contrastive loss of exposure contrastive pairs.

| Model                 | PSNR         | SSIM        | Colorfulness   | White Balance   | Exposure    |
|-----------------------|--------------|-------------|----------------|-----------------|-------------|
| Base                  | **22.154**   | **0.850**   | 0.489          | 0.354           | 0.548       |
| Color-Enhanced        | 20.421       | 0.822       | **0.682**      | 0.310           | 0.558       |
| WhiteBalance-Enhanced | 20.490       | 0.839       | 0.332          | **0.549**       | 0.559       |
| Exposure-Enhanced     | 21.202       | 0.830       | 0.407          | 0.230           | **0.618**   |
| All-Enhanced          | 18.985       | 0.771       | 0.622          | 0.704           | 0.573       |

The all-Enhanced model has the lowest PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index Measure) values among all models. This indicates that the all-Enhanced model's output images have the most difference from the original images in terms of these two metrics.

However, the all-Enhanced model has higher values in the Colorfulness, White Balance, and Exposure metrics compared to the Base model. This suggests that the all-Enhanced model is more effective in enhancing the colorfulness, white balance, and exposure of the images, despite having a lower fidelity to the original images (as indicated by the lower PSNR and SSIM).

It's important to note that a lower PSNR or SSIM doesn't necessarily mean the model is worse. It depends on the specific application and the trade-off between fidelity to the original image (PSNR, SSIM) and the enhancement of certain aspects (Colorfulness, White Balance, Exposure).

## Usage
- Download the [UIEB Dataset](https://li-chongyi.github.io/proj_benchmark.html) and place it in the `datasets` folder.
- Run `python train.py` to train the model.

## Pretrained Models
Download the pretrained models and place them in the `weights` folder.
- [Base Model](https://www.dropbox.com/s/0nzt1jowxavbkwa/replicated-waternet-20220528.pt?dl=0)
- [Color-Enhanced Model](https://www.dropbox.com/scl/fi/qqahjhajo94jflv8e9u4e/color-enhanced.pt)
- [WhiteBalance-Enhanced Model](https://www.dropbox.com/scl/fi/p358ov2hgxq4cvr3iwjjf/wb-enhanced.pt)
- [Exposure-Enhanced Model](https://www.dropbox.com/scl/fi/ubl8it9fgihnsj1c4fcw1/expo-enhanced.pt)
- [All-Enhanced Model](https://www.dropbox.com/scl/fi/rrija3y3ch78tedjdvnr7/all-enhanced.pt)

## Results and Demo
- `notebooks/3.inference_data_analysis.ipynb` contains the code for generating the raw results table.
- `notebooks/4.contrastive_pair_comparison_analysis.ipynb` contains the code for generating the contrastive pair comparison table.
- `notebooks/4.image_generationd.ipynb` contains the code for generating the images in the results table.
- `notebooks/5.demo.ipynb` contains the demo code for the model.

## Examples
![Comparison Image](./results/comparisons.png)

## Demo Video
- [video link](https://drive.google.com/file/d/1_e7BWBWpXRieAXthauFQF2wVLW1-cCsj/view?usp=drive_link)

## References
- [WaterNet](https://github.com/tnwei/waternet)
- [UIEB](https://li-chongyi.github.io/proj_benchmark.html)
- [LSUI](https://lintaopeng.github.io/code/)
- [CLIP](https://github.com/openai/CLIP)