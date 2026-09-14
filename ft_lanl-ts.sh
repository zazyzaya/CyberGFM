python pretrain.py --dirty --device 2 --log-out pretrained/static/lanl14argus-dirtyts/

python lp_finetune.py --dirty --walk-len 2 --device 2
python lp_finetune.py --dirty --walk-len 4 --device 2
python lp_finetune.py --dirty --walk-len 6 --device 2
python lp_finetune.py --dirty --walk-len 8 --device 2
python lp_finetune.py --dirty --walk-len 10 --device 2
python lp_finetune.py --dirty --walk-len 16 --device 2
python lp_finetune.py --dirty --walk-len 32 --device 2

python lp_finetune.py --dirty --walk-len 2 --device 2 --best
python lp_finetune.py --dirty --walk-len 4 --device 2 --best
python lp_finetune.py --dirty --walk-len 6 --device 2 --best
python lp_finetune.py --dirty --walk-len 8 --device 2 --best
python lp_finetune.py --dirty --walk-len 10 --device 2 --best
python lp_finetune.py --dirty --walk-len 16 --device 2 --best
python lp_finetune.py --dirty --walk-len 32 --device 2 --best