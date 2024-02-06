from funct_feature_engenirin import *
from pipeline import *
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error,r2_score,make_scorer
from sklearn.model_selection import cross_val_score,cross_validate,GridSearchCV
import xgboost as xgb

from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib as plt

class Model:
    def __init__(self, X, y_train):
        self.X_train = X
        self.y_train = y_train
        self.preprocessed = False
        self.ridge_done= False
        self.rf_done =False
        self.XGboost_done=False
        self.stacking_done = False
    def preprocessing(self):
        if not self.preprocessed:
            self.X_train, self.pipeline = pipeline(self.X_train, self.y_train)
            self.preprocessed = True
        else:
            print("Preprocessing has already been done.")
    
    def linear_regression(self):
        if not self.preprocessed:
            self.preprocessing()
        self.lin_reg = LinearRegression()
        self.lin_reg.fit(self.X_train, self.y_train)
        y_pred = self.lin_reg.predict(self.X_train)
        R_2 = r2_score(self.y_train, y_pred)
        RMSE =root_mean_squared_error(self.y_train, y_pred)
        return R_2, RMSE
    def ridge(self):
        print("Ridge regression started")
        if not self.preprocessed:
            self.preprocessing()
        self.ridge_model = Ridge()
        self.ridge_model.fit(self.X_train, self.y_train)
        y_pred = self.ridge_model.predict(self.X_train)
        R_2 = r2_score(self.y_train, y_pred)
        RMSE = root_mean_squared_error(self.y_train, y_pred)
        self.ridge_done = True
        self.ridge_result=RMSE
        print("Ridge regression done")
        return R_2, RMSE
    def random_forest(self,Grid_search=True,param_grid={}):
        print("Rf regression started")
        rf = RandomForestRegressor(random_state=42)
        if Grid_search==True:
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_root_mean_squared_error')
            grid_search.fit(self.X_train, self.y_train)
            self.best_params_rf = grid_search.best_params_
            self.best_score_rf = -grid_search.best_score_
            self.rf_best_model= grid_search.best_estimator_
            ###saturat,saturacio=check_grid(param_grid,self.best_params_rf)
            ##if(saturat):
               ## print("El random forest està saturat a ",saturacio)
        
        else:
            scoring = {'r2': make_scorer(r2_score), 'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(root_mean_squared_error(y_true, y_pred)))}


            scores = cross_validate(rf, self.X_train, self.y_train, cv=5, scoring=scoring)

            r2_scores = scores['test_r2']
            rmse_scores = scores['test_rmse']
            print("R2",r2_scores,"rmse:",rmse_scores)
        self.rf_done=True
        print("Rf regression done")
    def gb_boost(self,Grid_search=True,param_grid={}):
        print("Xgboost regression started")
        gb = xgb.XGBRegressor()
        if Grid_search==True:
            grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_root_mean_squared_error')
            grid_search.fit(self.X_train, self.y_train)
            self.best_params_gb = grid_search.best_params_
            self.best_score_gb = -grid_search.best_score_
            self.gb_best_model= grid_search.best_estimator_
            ##saturat,saturacio=check_grid(param_grid,self.best_params_rf)
            ##if(saturat):
               ## print("El XgBoost està saturat a ",saturacio)
        else:
            scoring = {'r2': make_scorer(r2_score), 'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(root_mean_squared_error(y_true, y_pred)))}
            scores = cross_validate(gb, self.X_train, self.y_train, cv=5, scoring=scoring)

            r2_scores = scores['test_r2']
            rmse_scores = scores['test_rmse']
            print("R2",r2_scores,"rmse:",rmse_scores)
        self.XGboost_done=True
        print("Xgboost regression done")

    def stacking(self):
        estimators = [
            ('ridge', Ridge()),
            ('rf', self.rf_best_model),
            ('gb', self.gb_best_model)
        ]
        stacking_regressor = StackingRegressor(
            estimators=estimators,
            final_estimator=MLPRegressor(random_state=42),
            cv=5
        )

        # Entrenar el model d'apilament
        score = cross_val_score(stacking_regressor, self.X_train, self.y_train, cv=5, scoring='neg_root_mean_squared_error')
        self.stacking_score = -np.mean(score)

        self.stacking_model = stacking_regressor.fit(self.X_train,self.y_train)
        self.stacking_done=True
    def best_model_selection(self):
        # Train models if not done yet
        if not self.ridge_done:
            self.ridge()
        if not self.rf_done:
            self.random_forest(param_grid={'n_estimators': [100, 250, 500], 'max_depth': [3, 5, 7]})
        if not self.XGboost_done:
            self.gb_boost(param_grid={'n_estimators': [100, 250, 500], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]})
        if not self.stacking_done:
            self.stacking()
        # Initialize a dictionary to hold model scores and references
        model_scores = {
            "Ridge": (self.ridge_result, self.ridge_model),
            "Random Forest": (self.best_score_rf, self.rf_best_model),
            "XGBoost": (self.best_score_gb, self.gb_best_model),
            "Stacking": (self.stacking_score, self.stacking_model)  # Assuming there's a stacking_model attribute
        }
        # Find the best model based on the score
        best_model_str, (self.best_result, self.best_model) = min(model_scores.items(), key=lambda x: x[1][0])

        print(f"The best model is {best_model_str} with a score of {self.best_result}")


    def test(self,X_test,y_test):

        transformed_data=self.pipeline.transform(X_test)
        numeric_features = X_test.select_dtypes(include=['number']).columns
        categorical_features = X_test.select_dtypes(exclude=['number']).columns
        new_column_names = np.concatenate((numeric_features,categorical_features)) # Adjust as needed
        X_test = pd.DataFrame(transformed_data, columns=new_column_names)

        y_pred = self.best_model.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)
        
        print("RMSE: ", rmse)
        graphic(y_test,y_pred)
    def coef_ridge(self):
        coeficients = self.ridge_model.coef_
        self.coeficients_ridge ={}
        for i in range(len(self.X_train.columns)):
            self.coeficients_ridge[self.X_train.columns[i]]=abs(coeficients[i])

    def coef_rf(self):
        importances = self.best_model.feature_importances_
        self.importances_rf ={}
        for i in range(len(self.X_train.columns)):
            self.importances_rf[self.X_train.columns[i]]=importances[i]
    def coef_gb(self):
        importances = self.best_model.feature_importances_
        self.importances_gb ={}
        for i in range(len(self.X_train.columns)):
            self.importances_gb[self.X_train.columns[i]]=importances[i]
    def worst_feat_model(self):
        self.coef_rf()
        self.coef_gb()
        self.coef_ridge
        print("Features menys importants al random forest")
        show_worst_features(self.importances_rf, n=5)
        print("Features menys importants al gradient boosting")
        show_worst_features(self.importances_gb, n=5)
        print("Features menys importants al gradient boosting")
        show_worst_features(self.coeficients_ridge, n=5)

def graphic(y_true,y_pred):
    plt.scatter(y_true,y_pred)

    plt.legend()
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--') # 'k--' crea una línia discontínua negra

    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    plt.title('Test predictions')
def show_worst_features(coefficients, n):
    sorted_features = sorted(coefficients.items(), key=lambda x: x[1])
    worst_n_features = sorted_features[:n]

    print(f"The {n} features with the smallest coefficients are:")
    for feature, coef in worst_n_features:
        print(f"{feature}: {coef}")

def graphic_colors(y_true_bar,y_pred_bar,y_true_car, y_pred_car):
    plt.scatter(y_true_bar, y_pred_bar, color='blue')
    plt.scatter(y_true_car, y_pred_car, color='red')

    plt.legend()
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    plt.plot([min(y_true_bar), max(y_true_car)], [min(y_true_bar), max(y_true_car)], 'k--') # 'k--' crea una línia discontínua negra

    plt.title('Test predictions')
## TO DO ##
    """
    Function to check if the hiperameters are overflowed
    """
###