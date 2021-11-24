import rfclass
import pandas as pd

if __name__ == '__main__':
    params = {
        'n_trees': 100,
        'max_depth': 5,
        'n_batches_per_layer': 1,
        # You must enable center_bias = True to get DFCs. This will force the model to
        # make an initial prediction before using any features (e.g. use the mean of
        # the training labels for regression or log odds for classification when
        # using cross entropy loss).
        'center_bias': True
    }
    data = pd.read_csv('heart.csv')
    features_category = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']
    features_numeric = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
    target = 'output'

    rfclassifier = rfclass.rf_class(data,target,features_category,features_numeric,params)

    train_input_fn = rfclass.make_input_fn(rfclassifier.train, rfclassifier.y_train, NUM_EXAMPLES=rfclassifier.NUM_EXAMPLES)
    rfclassifier.rf_classifier.train(train_input_fn, max_steps=1000)

    eval_input_fn = rfclass.make_input_fn(rfclassifier.train, rfclassifier.y_train, shuffle=False, n_epochs=1,NUM_EXAMPLES=rfclassifier.NUM_EXAMPLES)
    result = rfclassifier.rf_classifier.evaluate(eval_input_fn)
    print(pd.Series(result))