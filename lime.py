import lime
import lime.lime_tabular
import numpy as np
from sklearn.linear_model import RandomForestRegressor
def lime_explainer_1(X,y):
# Initialize the LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X),  # Use training data here
        feature_names=X.columns,  # Make sure X_train has columns attribute
        class_names=['Price'],
        mode='regression'
    )
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest_model.fit(X, y)
    num_features = 30
    # Choose a specific instance to explain
    idx = np.where(y > 350000)[0]
    if len(idx) > 0:
        instance_index = idx[0]  # Choose the first instance for simplicity
        instance = X.iloc[instance_index]

        # Generate explanation for the chosen instance
        exp = explainer.explain_instance(
            data_row=instance.values,  # Convert to numpy array
            predict_fn=random_forest_model.predict,  # Make sure reg is your trained model
            num_features=num_features
        )

        # Displaying the explanation
        fig = exp.as_pyplot_figure()
        fig.show()
    else:
        print("No instances found with y_train > 350000")

def lime_2(X,y,model_used):

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X),  # Use training data here
        feature_names=X.columns,  # Make sure X_train has columns attribute
        class_names=['Price'],
        mode='regression'
    )

    num_features = 20
    # Choose a specific instance to explain
    idx = np.where(y > 400000)[0]
    if len(idx) > 0:
        instance_index = idx[0]  # Choose the first instance for simplicity
        instance = X.iloc[instance_index]

        # Generate explanation for the chosen instance
        exp = explainer.explain_instance(
            data_row=instance.values,  # Convert to numpy array
            predict_fn=model_used.predict,  # Make sure reg is your trained model
            num_features=num_features
        )

        # Displaying the explanation
        fig = exp.as_pyplot_figure()
        fig.show()
    else:
        print("No instances found with y_train > 350000")

