
'''
Load necessary libraries
'''
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Create a portable function for manipulating the datasets


def data_processing(df1, df2, modelname, outputfile):
    '''
    This function takes in two datasets.
    Reads in individual datasets and sorts each one by the title column.
    Extracts the decision column from df2 and adds it to df1 with a user-supplied column name.
    Returns a reference dataframe with the model decision column to be passed to a confusion matrix for visualization.

    Parameters:
    df1 - Reference dataset, with human decision.
    df2 - Test dataset (model output) decision.
    modelname - Name of the model (e.g., 'zero_shot', 'one_shot', 'reasoning', etc.).
    outputfile - Name of the output CSV file.
    '''
    # Load datasets
    human_decision = pd.read_csv(df1)
    LLM_decision = pd.read_csv(df2)

    # Sort datasets by title column
    human_decision_sorted = human_decision.sort_values(by='title')  
    LLM_decision_sorted = LLM_decision.sort_values(by='title')  

    # Add model decision column using the provided model name
    human_decision_sorted[modelname] = LLM_decision_sorted['decision']

    # Print preview of updated dataset
    print(human_decision_sorted.head())

    # Save the updated dataset to the specified output file
    human_decision_sorted.to_csv(outputfile, index=False)

    return human_decision_sorted  # Return the updated DataFrame


#  Create a function to compute the confusion matrix capturing TP,TN, FP and FN and visualise using heatmap
# def plot_confusion_matrix(human_labels, model_labels, model_name):
#     '''
#     A function to compute the confusion matrix comparing human labels
#     to LLM labels.
#     Further this function visualises the matrix using a heatmap showing
#     TP, TN, FP and FN

#     Parameters:
#     human_labels - human decision (2nd reviewer)
#     model_labels - LLM decision (decision by model)
#     '''
#     # Compute confusion matrix
#     cm = confusion_matrix(human_labels, model_labels, labels=["Included", "Excluded"])
    
#     # Plotting confusion matrix heatmap
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Included", "Excluded"], yticklabels=["Included", "Excluded"])
#     plt.title(f"Confusion Matrix for {model_name}")
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.show()


def plot_confusion_matrix(human_labels, model_labels, model_name):
        '''
    A function to compute the confusion matrix comparing human labels
    to LLM labels.
    Further this function visualises the matrix using a heatmap showing
    TP, TN, FP and FN

    Parameters:
    human_labels - human decision (2nd reviewer)
    model_labels - LLM decision (decision by model)
    '''
    # Compute confusion matrix
    cm = confusion_matrix(human_labels, model_labels, labels=["Included", "Excluded"])
    
    # Extract TP, TN, FP, FN
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(human_labels, model_labels, pos_label="Included")

    # Plot confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Included", "Excluded"], yticklabels=["Included", "Excluded"])
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Print metrics
    print(f"Metrics for {model_name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("-" * 50)

# Compare each model with Human labels and plot the confusion matrix
for model in df.columns[3:]:  # Skip the first three columns (title, abstract, and human decision)
    plot_confusion_matrix(df['Human'], df[model], model)
    

