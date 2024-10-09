for i in {0..10}
do
   python3 infer_all_pairs.py --cuda 3 --fold $i 
done