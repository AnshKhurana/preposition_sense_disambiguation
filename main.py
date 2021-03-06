"""Main caller script."""
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from data import prep_feats, gen_train_test

preposition_list = ['about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'as', 'at', 'before', 'behind', 'beneath', 'beside', 'between', 'by', 'down', 'during', 'for', 'from', 'in', 'inside', 'into', 'like', 'of', 'off', 'on', 'onto', 'over', 'round', 'through', 'to', 'towards', 'with']

parser = argparse.ArgumentParser()

parser.add_argument('--window_size', type=int, default=2)
parser.add_argument('--feature_comb', type=str, default='concat', choices=['concat', 'linear'])
parser.add_argument('--preposition', type=str, default='run_all')
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--model', type=str, required=True, choices=['svm', 'knn', 'mlp'])
parser.add_argument('--knn_neighbours', type=int, default=8)
parser.add_argument('--svm_regularizer', type=float, default=1.0)
parser.add_argument('--hidden_neurons', type=int, default=20)
parser.add_argument('--save_test_outs', type=str, default='myout')

def check_and_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_preposition_data(train, test, feature_comb='concat', beta=None, gamma=None):
    
    Y_train = np.array(train['labels'])
    ftrain = train['sents']
    ftest = test['sents']
    feat_keys = ['vl', 'vr', 'vi']
    train_feats = dict()
    test_feats = dict()

    train_sentences = np.array([sent['sentence'] for sent in ftrain])
    test_sentences = np.array([sent['sentence'] for sent in ftest])

    for feat_key in feat_keys:
        train_feats[feat_key] = np.array([feat[feat_key] for feat in ftrain])
        test_feats[feat_key] = np.array([feat[feat_key] for feat in ftest])

    
    if feature_comb == 'concat':
        X_train = np.concatenate((train_feats['vr'], train_feats['vl'], train_feats['vi']), axis=1)
        X_test = np.concatenate((test_feats['vr'], test_feats['vl'], test_feats['vi']), axis=1)
    elif feature_comb == 'linear':
        X_train = train_feats['vl'] + beta * train_feats['vr'] + gamma * train_feats['vi']
        X_test = test_feats['vl'] + beta * test_feats['vr'] + gamma * test_feats['vi']
    else:
        raise NotImplementedError('Feature combination %s is not implemented.' % feature_comb)

    print('X_train: ', X_train.shape)
    print('X_test: ', X_test.shape)
    print('Y_train: ', Y_train.shape)

    return X_train, X_test, np.squeeze(Y_train), train_sentences, test_sentences

def run_preposition(preposition, train, test, feature_comb, output_dir,
                    beta=1, gamma=1, model='svm', svm_regularizer=1.0, knn_neighbours=8, hidden_neurons=20):
    X_trainset, X_test, Y_trainset, train_sentences, test_sentences = process_preposition_data(train, test, feature_comb, beta, gamma)

    print('Training model for preposition: %s' % preposition)

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_trainset, Y_trainset, shuffle=True, random_state=42, test_size=0.10)

    if model == 'svm':
        # default parameters
        classifier = SVC(C=svm_regularizer)
    elif model == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=knn_neighbours)
    elif model == 'mlp':
        classifier = MLPClassifier(hidden_layer_sizes=(hidden_neurons))
    else:
        raise NotImplementedError('Algorithm %s has not been implemented.')

    classifier.fit(X_train, y_train)

    # Evaluate on validation data 
    y_pred_train = classifier.predict(X_train)
    y_pred_val = classifier.predict(X_val)
    y_pred_test = classifier.predict(X_test)

    print('Test labels')
    print(y_pred_test)

    print('Training Accuracy')
    print(classification_report(y_train, y_pred_train))
    print('Validation Accuracy')
    print(classification_report(y_val, y_pred_val))

    # save test outputs

    check_and_make_dir(output_dir)

    with open(os.path.join(output_dir, '%s.out' % preposition), 'w') as f:
        for sent, label in zip(test_sentences, y_pred_test):
            f.write('%s | %s\n' % (sent, label))

def run_all_preps(trainset, testset, feature_comb, output_dir,
                    beta=1, gamma=1, model='svm', svm_regularizer=1.0, knn_neighbours=8, hidden_neurons=20):
    
    for preposition in preposition_list:
        prep_train = trainset[preposition]
        prep_test = testset[preposition]
        try:
            run_preposition(preposition, prep_train, prep_test, feature_comb,
            output_dir, beta, gamma, model, svm_regularizer, knn_neighbours, hidden_neurons)
        except:
            continue


if __name__ == '__main__':
    args = parser.parse_args()
    print('Preparing dataset...')
    train, test = gen_train_test(args.window_size)
    preposition = args.preposition
    if preposition == 'run_all':
        run_all_preps(train, test, args.feature_comb, args.save_test_outs, 
        args.beta, args.gamma, args.model, args.svm_regularizer, args.knn_neighbours, args.hidden_neurons)
    elif preposition in preposition_list:
        run_preposition(preposition, train[preposition], test[preposition], args.feature_comb, args.save_test_outs, 
        args.beta, args.gamma, args.model, args.svm_regularizer, args.knn_neighbours, args.hidden_neurons)
    else:
        raise ValueError('Preposition %s is not defined.' % preposition)
    