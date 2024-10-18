# -*- coding: UTF-8 -*-

import sys
import getopt
import tensorflow as tf
import os
import re

"""
future works:
use a keyword to filter which vars need add or remove prefix
merge two or more ckpts
"""

usage_str = '''
USAGE:
python tensorflow_rename_variables_tf2.py --checkpoint_dir=path/to/dir/ 
            --mode=replace or remove or add_prefix or check
            --replace_from=substr --replace_to=substr
            --keyword=substr --prefix=abc --dry_run

EXAMPLES:
replace: python tensorflow_rename_variables_tf2.py --checkpoint_dir=./best_baseline_0 --mode=replace --replace_from=pointnet --replace_to=removed_pointnet --dry_run
add_prefix: python tensorflow_rename_variables_tf2.py --checkpoint_dir=./best_baseline_0 --mode=add_prefix --prefix=parts/ --dry_run
remove: python tensorflow_rename_variables_tf2.py --checkpoint_dir=./best_baseline_0 --mode=remove --keyword=ssg-layer2 --dry_run
check: python tensorflow_rename_variables_tf2.py --checkpoint_dir=./output_models --mode=check
'''

def check(checkpoint_dir):
    variables = tf.train.list_variables(checkpoint_dir)
    for var_name, shape in variables:
        print('%s' % (var_name))


def is_valid_variable_name(name):
    # Variable names must match the regex pattern
    pattern = r'^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$'
    return re.match(pattern, name) is not None


def replace(checkpoint_dir, replace_from, replace_to, dry_run):
    assert None not in [replace_from, replace_to], 'must specify replace_from and replace_to'
    variables = tf.train.list_variables(checkpoint_dir)
    new_variables = {}
    for var_name, shape in variables:
        if var_name.startswith('_'):
            print(f"Skipping internal variable: {var_name}")
            continue
        if not is_valid_variable_name(var_name):
            print(f"Skipping invalid variable name: {var_name}")
            continue
        # Load the variable
        value = tf.train.load_variable(checkpoint_dir, var_name)

        # Set the new name
        new_name = var_name.replace(replace_from, replace_to)

        print('%-50s ==> %-50s' % (var_name, new_name))
        new_variables[new_name] = value

    if not dry_run:
        with tf.Graph().as_default():
            # Create variables with the new names
            new_vars = {}
            for name, value in new_variables.items():
                if not is_valid_variable_name(name):
                    print(f"Skipping variable with invalid new name: {name}")
                    continue
                var = tf.compat.v1.get_variable(name=name, initializer=value)
                new_vars[name] = var

            saver = tf.compat.v1.train.Saver(var_list=new_vars)
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                save_path = os.path.join('./new_policy', 'ckpt-255000')
                saver.save(sess, save_path, write_meta_graph=False)  # Prevent saving the .meta file
                print(f'Model saved to {save_path}')


def remove(checkpoint_dir, keyword, dry_run):
    assert keyword is not None, 'must specify keyword'
    variables = tf.train.list_variables(checkpoint_dir)
    new_variables = {}
    for var_name, shape in variables:
        if var_name.startswith('_'):
            print(f"Skipping internal variable: {var_name}")
            continue
        if not is_valid_variable_name(var_name):
            print(f"Skipping invalid variable name: {var_name}")
            continue
        if keyword not in var_name:  # to save
            # Load the variable
            print('save: %s' % (var_name))
            value = tf.train.load_variable(checkpoint_dir, var_name)
            new_variables[var_name] = value
        else:  # to remove
            print('remove: %s' % (var_name))

    if not dry_run:
        with tf.Graph().as_default():
            # Create variables with the retained names
            new_vars = {}
            for name, value in new_variables.items():
                if not is_valid_variable_name(name):
                    print(f"Skipping variable with invalid name: {name}")
                    continue
                var = tf.compat.v1.get_variable(name=name, initializer=value)
                new_vars[name] = var

            saver = tf.compat.v1.train.Saver(var_list=new_vars)
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                save_path = os.path.join('./output_models', 'model')
                saver.save(sess, save_path, write_meta_graph=False)  # Prevent saving the .meta file
                print(f'Model saved to {save_path}')


def add_prefix(checkpoint_dir, prefix, dry_run):
    assert prefix is not None, 'must specify prefix'
    variables = tf.train.list_variables(checkpoint_dir)
    new_variables = {}
    for var_name, shape in variables:
        if var_name.startswith('_'):
            print(f"Skipping internal variable: {var_name}")
            continue
        if not is_valid_variable_name(var_name):
            print(f"Skipping invalid variable name: {var_name}")
            continue
        # Load the variable
        value = tf.train.load_variable(checkpoint_dir, var_name)

        # Set the new name
        new_name = prefix + var_name

        print('%-50s ==> %-50s' % (var_name, new_name))
        new_variables[new_name] = value

    if not dry_run:
        with tf.Graph().as_default():
            # Create variables with the new names
            new_vars = {}
            for name, value in new_variables.items():
                if not is_valid_variable_name(name):
                    print(f"Skipping variable with invalid new name: {name}")
                    continue
                var = tf.compat.v1.get_variable(name=name, initializer=value)
                new_vars[name] = var

            saver = tf.compat.v1.train.Saver(var_list=new_vars)
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                save_path = os.path.join('./output_models', 'model')
                saver.save(sess, save_path, write_meta_graph=False)  # Prevent saving the .meta file
                print(f'Model saved to {save_path}')


def main(argv):
    checkpoint_dir = None
    mode = None
    replace_from = None
    replace_to = None
    prefix = None
    keyword = None
    dry_run = False

    try:
        opts, args = getopt.getopt(argv, 'h', ['help=', 'checkpoint_dir=', 'mode=',
                                  'replace_from=', 'replace_to=', 'prefix=', 'keyword=', 'dry_run'])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt == '--checkpoint_dir':
            checkpoint_dir = arg
        elif opt == '--mode':
            mode = arg
        elif opt == '--replace_from':
            replace_from = arg
        elif opt == '--replace_to':
            replace_to = arg
        elif opt == '--prefix':
            prefix = arg
        elif opt == '--keyword':
            keyword = arg
        elif opt == '--dry_run':
            dry_run = True

    if not checkpoint_dir:
        print('Please specify a checkpoint_dir. Usage:')
        print(usage_str)
        sys.exit(2)

    if mode not in ['replace', 'remove', 'add_prefix', 'check']:
        print('Please specify a valid mode. Usage:')
        print(usage_str)
        sys.exit(2)

    if dry_run:
        print('--------- dry run ---------')

    if mode == 'replace':
        replace(checkpoint_dir, replace_from, replace_to, dry_run)
    elif mode == 'remove':
        remove(checkpoint_dir, keyword, dry_run)
    elif mode == 'add_prefix':
        add_prefix(checkpoint_dir, prefix, dry_run)
    elif mode == 'check':
        check(checkpoint_dir)
    else:
        raise ValueError('Unknown mode: {}'.format(mode))


if __name__ == '__main__':
    main(sys.argv[1:])
