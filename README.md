# hf_namespace_analysis

Automatic Analysis Script for Huggingface Namespaces

## Installation

```shell
git clone https://github.com/deepghs/hf_namespace_analysis.git
cd hf_namespace_analysis
pip install -r requirements.txt
```

## Run Analysis

```shell
# export HF_TOKEN=your_hf_token
python -m analysis.run -a narugo 
```

The analysis result will be published at dataset repository narugo/storage_analysis.

If you need to specify the repository to save the result, use `-r` to explicitly assign that.

If you need to analysis the private repositories, use `--private`.

