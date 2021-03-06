import argparse
import os

preposition_list = ['about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'as', 'at', 'before', 'behind', 'beneath', 'beside', 'between', 'by', 'down', 'during', 'for', 'from', 'in', 'inside', 'into', 'like', 'of', 'off', 'on', 'onto', 'over', 'round', 'through', 'to', 'towards', 'with']


parser = argparse.ArgumentParser()

parser.add_argument('--my_out', type=str)

args = parser.parse_args()


for prep in preposition_list:
    reference_file = os.path.join('reference_test_out', '%s.out' % prep)
    my_out_file = os.path.join(args.my_out, '%s.out' % prep)

    with open(reference_file, 'r') as f:
        num_lines_ref = len(f.readlines())

    with open(my_out_file, 'r') as f:
        num_lines_my = len(f.readlines())
    
    if num_lines_my != num_lines_ref:
        print('mismatch ', prep)
    else:
        print('matched! ', prep)


