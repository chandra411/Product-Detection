CUDA_VISIBLE_DEVICES=0 python inference.py --gt=./data/annotation.txt --test_dir=../GroceryDataset_part1/Shelfimages/test/ --pb=./test_pb/frozen_inference_graph.pb --out_dir=./test_out --save_im=0
