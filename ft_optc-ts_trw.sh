python cls_finetune.py --optc-argus --walk-len 2 --device 2 --trw --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 4 --device 2 --trw --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 6 --device 2 --trw --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 8 --device 2 --trw --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 10 --device 2 --trw --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 16 --device 2 --trw --epochs 5 --tag _unfrozen
python cls_finetune.py --optc-argus --walk-len 32 --device 2 --trw --epochs 5 --tag _unfrozen

python cls_finetune.py --trw --optc-argus --walk-len 2 --device 2 --model-fname pretrained/temporal/optc-argus/trw_bert_optc-argus_tiny-best.pt --tag _best --epochs 5 --tag _unfrozen
python cls_finetune.py --trw --optc-argus --walk-len 4 --device 2 --model-fname pretrained/temporal/optc-argus/trw_bert_optc-argus_tiny-best.pt --tag _best --epochs 5 --tag _unfrozen
python cls_finetune.py --trw --optc-argus --walk-len 6 --device 2 --model-fname pretrained/temporal/optc-argus/trw_bert_optc-argus_tiny-best.pt --tag _best --epochs 5 --tag _unfrozen
python cls_finetune.py --trw --optc-argus --walk-len 8 --device 2 --model-fname pretrained/temporal/optc-argus/trw_bert_optc-argus_tiny-best.pt --tag _best --epochs 5 --tag _unfrozen
python cls_finetune.py --trw --optc-argus --walk-len 10 --device 2 --model-fname pretrained/temporal/optc-argus/trw_bert_optc-argus_tiny-best.pt --tag _best --epochs 5 --tag _unfrozen
python cls_finetune.py --trw --optc-argus --walk-len 16 --device 2 --model-fname pretrained/temporal/optc-argus/trw_bert_optc-argus_tiny-best.pt --tag _best --epochs 5 --tag _unfrozen
python cls_finetune.py --trw --optc-argus --walk-len 32 --device 2 --model-fname pretrained/temporal/optc-argus/trw_bert_optc-argus_tiny-best.pt --tag _best --epochs 5 --tag _unfrozen