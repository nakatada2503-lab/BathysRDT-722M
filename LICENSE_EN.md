# BathysRDT Research License — English reference translation

Version 1.0 (2026-10-02)

**This is a reference translation. The Japanese text in `LICENSE` governs. If the two differ, the Japanese text prevails.**

Copyright (c) 2026 Tadayuki Nakamura. All rights reserved except as expressly granted below.

## 1. Definitions
1. "Licensor" means Tadayuki Nakamura.
2. "User" means an individual or organization that agrees to this License and uses the Materials.
3. "Materials" means: (a) **Code and Documents** — source code (including inference code), configuration files, inference settings and documents; and (b) **Original Weights** — model weights distributed by the Licensor whose checkpoint names are designated by the Licensor on the distribution page.
4. Weights obtained by transforming the Original Weights **without training** (including format conversion, quantization, pruning and merging with other weights) are treated, **in their entirety**, as Original Weights.
5. "Derived Weights" means weights obtained by the User by further training the Original Weights (e.g., fine-tuning), thereby changing their parameters through training. Weights covered by item 4 are not Derived Weights.
6. Weights of a separate model trained only on the outputs of this model, and containing none of the parameters of the Original Weights, are neither Original Weights nor Derived Weights.
7. "Non-commercial research and educational purposes" means academic research, technical evaluation, education and personal learning, judged by **the purpose of the act of use**, regardless of whether the User's organization is for-profit. Excluded: (a) integration into, or operation for providing, products or services; (b) paid provision of the Materials, Derived Weights or functions using them; (c) contracted development or evaluation for consideration; (d) any other use whose main purpose is generating revenue.

## 2. Grant
Subject to this License, for non-commercial research and educational purposes only, the Licensor grants the User a non-exclusive, non-transferable, royalty-free right to:
1. use, reproduce and modify the **Code and Documents**;
2. use and reproduce (within the User's organization only) the **Original Weights**;
3. create, use and reproduce **Derived Weights** from the Original Weights.

## 3. Redistribution
1. **Code and Documents** may be redistributed, modified or not, only if (a) the full text of this License and the copyright notice are included, (b) modifications are clearly marked, and (c) they are provided under this same License.
2. **Original Weights** (including anything treated as Original Weights under 1.4) may not be redistributed, except for sharing within the User's organization for permitted purposes. Original Weights must be obtained from the distribution point designated by the Licensor.
3. **Derived Weights** may be published or redistributed only if (a) provided under this same License (non-commercial research and educational use only; no patent license), (b) the full text of this License and the copyright notice are included, (c) it is stated that they were created from the Licensor's Original Weights, together with the source checkpoint name (as designated by the Licensor) and a summary of the changes, and (d) they do not include the Original Weights or anything treated as Original Weights under 1.4.

## 4. Citation
When publishing results obtained using the Materials or Derived Weights, cite: Tadayuki Nakamura, "精度差分観測による重み共有再帰Transformerの学習ダイナミクス解明: Gate Decay振幅制御、FP32崩壊境界、および多制御器協調学習", Jxiv, 2026. DOI: 10.51094/jxiv.6476. This applies regardless of redistribution.

## 5. Patents and other rights
1. This License does not expressly grant any license under any patents, rights to obtain patents, utility model rights or other industrial property rights that the Licensor holds now or in the future.
2. Neither the provision of the Materials nor the wording of this License shall be construed as granting any license under such rights, whether by implication, estoppel or otherwise.
3. All rights not expressly granted in Section 2 are reserved by the Licensor.

## 6. Commercial use
Any use other than for non-commercial research and educational purposes (including commercial use) requires a separate prior written agreement with the Licensor.

## 7. Third-party materials
Third-party materials needed to use the Materials (tokenizer, teacher model, training data, etc.) are subject to their providers' terms. The User is responsible for checking and complying with them.

## 8. No warranty; limitation of liability
1. The Materials are provided "AS IS", without warranty of any kind, express or implied, including merchantability, fitness for a particular purpose, accuracy and non-infringement.
2. To the maximum extent permitted by law, the Licensor is not liable for any damages arising from the use of, or inability to use, the Materials.
3. Model outputs may contain errors or inappropriate content.

## 9. Termination
1. If the User breaches this License, the Licensor will notify the User. If the breach is not cured within 30 days of the notice, the User's rights under this License terminate.
2. Notwithstanding item 1, the User's rights terminate immediately upon (a) redistribution of Original Weights in breach of 3.2, or (b) use other than for non-commercial research and educational purposes in breach of Section 6.
3. Upon termination, the User shall stop using and delete the Materials, Derived Weights and all copies.
4. Termination of one User's rights does not affect the rights of third parties who lawfully received Derived Weights or Code and Documents under this License before the termination, **as long as those third parties comply with this License**.

## 10. Governing law and jurisdiction
This License is governed by the laws of Japan. The Yamagata District Court has exclusive jurisdiction in the first instance over any dispute relating to this License.

## 11. Language
The Japanese text governs. If this reference translation differs from the Japanese text, the Japanese text prevails.

## 12. Contact
License and commercial-use inquiries: Tadayuki Nakamura <tada2503@yahoo.co.jp>
