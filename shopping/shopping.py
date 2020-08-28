import csv
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    df = pd.read_csv('shopping.csv')
    df['Month'] = df['Month'].apply(switch_months).fillna(0).astype('int64')
    df['VisitorType'] = df['VisitorType'].apply(returning_visitor).astype('int64')
    df['Weekend'] = df['Weekend'].apply(weekend).astype('int64')
    df['Revenue'] = df['Revenue'].apply(revenue).astype('int64')
    return df.iloc[1:, :17], df.iloc[1:, 17]



def switch_months(argument):
    switcher = {
        "Jan": int(0),
        "Feb": int(1),
        "Mar": int(2),
        "Apr": int(3),
        "May": int(4),
        "Jun": int(5),
        "Jul": int(6),
        "Aug": int(7),
        "Sep": int(8),
        "Oct": int(9),
        "Nov": int(10),
        "Dec": int(11)
    }
    return switcher.get(argument)

def returning_visitor(argument):
    if argument == "Returning_Visitor":
        return 1
    else:
        return 0

def weekend(argument):
    if argument == True:
        return 1
    else:
        return 0

def revenue(argument):
    if argument == True:
        return 1
    else:
        return 0


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    #sensitivity tp/(tp + fn)
    #specificity tn/(tn + fp)
    for actual, prediction in zip(labels, predictions):
        if actual == prediction:
            #true positive
            if actual == 1:
                tp += 1
            #true negative
            else:
                tn += 1
        else:
            if actual == 1:
                #false positives
                if prediction == 0:
                    fp += 1
            else:
                #false negatives
                fn += 1

    return (tp) / (tp + fn), (tn) / (tn + fp)

if __name__ == "__main__":
    main()
