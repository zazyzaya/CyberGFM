python cls_finetune.py --optc-argus --walk-len 2 --device 1 --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 4 --device 1 --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 6 --device 1 --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 8 --device 1 --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 10 --device 1 --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 16 --device 1 --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 32 --device 1 --epochs 5 --tag _unfrozen

python cls_finetune.py --optc-argus --walk-len 2 --device 1 --model-fname pretrained/static/optc-argus/rw_bert_optc-argus_tiny-best.pt --tag _best --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 4 --device 1 --model-fname pretrained/static/optc-argus/rw_bert_optc-argus_tiny-best.pt --tag _best --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 6 --device 1 --model-fname pretrained/static/optc-argus/rw_bert_optc-argus_tiny-best.pt --tag _best --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 8 --device 1 --model-fname pretrained/static/optc-argus/rw_bert_optc-argus_tiny-best.pt --tag _best --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 10 --device 1 --model-fname pretrained/static/optc-argus/rw_bert_optc-argus_tiny-best.pt --tag _best --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 16 --device 1 --epochs 5 --model-fname pretrained/static/optc-argus/rw_bert_optc-argus_tiny-best.pt --tag _best --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 32 --device 1 --epochs 5 --model-fname pretrained/static/optc-argus/rw_bert_optc-argus_tiny-best.pt --tag _best --epochs 5 --tag _unfrozen