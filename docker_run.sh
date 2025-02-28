conda activate watermarknn
python train.py --batch_size 100 --max_epochs 60 --runname train --wm_batch_size 2 --wmtrain
python predict.py --model_path checkpoint/model.t7
python fine-tune.py --lr 0.01 --load_path checkpoint/model.t7 --save_dir checkpoint/ --save_model ftll.t7 --runname fine.tune.last.layer
python fine-tune.py --lr 0.01 --load_path checkpoint/model.t7 --save_dir checkpoint/ --save_model ftal.t7 --runname fine.tune.all.layers --tunealllayers
python fine-tune.py --lr 0.01 --load_path checkpoint/model.t7 --save_dir checkpoint/ --save_model rtll.t7 --runname reinit.last.layer --reinitll
python fine-tune.py --lr 0.01 --load_path checkpoint/model.t7 --save_dir checkpoint/ --save_model rtal.t7 --runname reinit_all.layers --reinitll --tunealllayers