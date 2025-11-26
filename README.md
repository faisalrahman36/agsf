This is version 1 of Astronomy GMM Source Finder (AGSF).

Author: Syed Faisal ur Rahman

The core references used are :

Package: Scikit-learn
Algorithm: GMM (https://scikit-learn.org/stable/modules/mixture.html)


References:
1.  **Condon, J. J. (1997).** "Errors in Elliptical Gaussian Fits." *PASP*, 109, 166.
2.  **Dempster, A. P., et al. (1977).** "Maximum likelihood from incomplete data via the EM algorithm."
3.  **Schwarz, G. (1978).** "Estimating the dimension of a model." *The Annals of Statistics*.

Acknowledgements:

The author acknowledges the assistance of multiple large language models (LLMs) in the engineering and technical implementation of AGSF. Specifically, the Gemini (Google) and Claude (Anthropic) models were employed as coding assistants for tasks including refining complex Python syntax, debugging the parallel processing logic in joblib, and optimizing array handling within the tiled FITS processing framework. The use of these models was limited to improving the efficiency of the implementation with explicit instructions from author especially to deal with the coding edge cases. All core theoretical framework, algorithmic contributions, astrophysical methodology (including the GMM fitting, deconvolution, and error derivation), validation tests, and scientific conclusions remain the responsibility of the author.

If using this then please cite:

**Rahman, S. F. (2025).** *AGSF: A  Probabilistic Source Finding Pipeline for Radio Interferometry using Gaussian Mixture Model (GMM). (In preparation)*
And add link to this repo.

Please send feedback at: faisalrahman36@hotmail.com
