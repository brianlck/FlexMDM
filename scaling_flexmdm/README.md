<div  align="center">
    <h1> Scaling FlexMDM from LLaDA weight initialization</h1>
  <p> Scaling FlexMDM from LLaDA-8B-Base, an open-sourced pretrained MDM. Scaled FlexMDM outperforms LLaDA, on math (GSM8K) and code infilling (HumanEval-infill).  </p>
</div>


<div align="center">
  <hr width="100%">
</div>



## Overview

The code consists of three main parts: (1) Transfer learning from LLaDA-8B-Base, (2) Instruction-fine tuning the baseline models, and (3) Inference for both models and evaluation. Experimental details can be found in Appendix E of our paper. Our code is built from LLaDA-d1 [codebase](https://github.com/dllm-reasoning/d1).

## Transfer learning

We first take LLaDA-8B-Base and add a time-embedding / scalar insertion head (`llada_dit.py` and `llada_utils.py`). Next, we pre-process the transfer learning data with `flexmdm_transfer_preprocess.py`. The training code is given in `flexmdm_transfer_openwebtext.py`, for which the slurm script is `scripts/flexmdm_transfer.sh`. As we mentioned, our FlexMDM baseline is fine-tuned with 16 H100 GPUs for approximately three days. The FlexMDM interpolant is given in `flexmdm_interpolant.py`. `flexmdm_trainer.py` file includes the data collector as well as the loss function calculation.

## Instruction-Fine Tuning

We next instruction-fine-tune (IFT) both the FlexMDM (which we acquired from the transfer learning) and LLaDA-8B-Base. For the IFT with GSM8K, the data processing code is given in `preprocess_math.py`. For the IFT with the opc-sft dataset, it is given in `preprocess_code_infilling.py`. Both IFT shares the training code in `instruction_finetuning.py` and the training scripts are respectively given by `scripts/IFT_math.sh` and `scripts/IFT_code.sh`. `llada_trainer.py` file includes the data collector as well as the loss function calculation.

## Evaluation
Each model's inference implementation is given in `flexmdm_inference.py` and `llada_inference.py`, respectively. The evaluation code for GSM8K and HumanEval-infill is given in `eval_gsm8k.py` and `eval_humaneval_infill.py`, respectively. The slurm scripts are provided in `scripts/inference_math.sh` and `scripts/inference_code.sh`.