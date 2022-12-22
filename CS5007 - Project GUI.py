# Final Project CS 5007 - Team 4
# By Edith Gomez, Michael Da Silva and Gabriel Torres
# This is a curve fitting application that provides a catalog of curve fitting models so the user can upload their
# own CSV file and visualize different regression models outputs like error, accuracy and coefficients

# Requirements: the input file has to be in CSV format, only numerical variables are allowed as inputs,
# the CSV file has to be clean (no missing values), only one target variable is allowed.

from tkinter import *
from tkinter import ttk
from tkinter import font
import pandas as pd
from openpyxl.workbook import Workbook
from openpyxl import load_workbook
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# This function reads an input csv file and transforms it into a pandas dataframe
def import_data(filename, columns, separator):
    data = pd.read_csv(filename, sep=separator)
    data = data[columns]  # attributes
    return data


# This function extracts all the column names that the user inputs as independent variables
def get_all_columns(independent, target):
    ind_list = independent.split(",")
    dep_list = target.split(",")
    all_variables = ind_list + dep_list

    return all_variables


# This function fits a Linear Regression model to the input dataframe
def linearprediction(data, prediction):
    var_names = list(data.columns.values)
    var_names.remove(prediction)
    x = np.array(data.drop([prediction], 1))
    y = np.array(data[prediction])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    y_train_predict = linear.predict(x_train)
    y_test_predict = linear.predict(x_test)
    test_rmse = round((np.sqrt(mean_squared_error(y_test, y_test_predict))), 3)
    test_r2 = round(r2_score(y_test, y_test_predict), 3)
    coefficients = np.around(linear.coef_, decimals=3)
    intercept = np.around(linear.intercept_, decimals=3)

    output_dict = {'model': "Linear Regression", 'test_rmse': test_rmse,
                   'test_r2': test_r2, 'Independent variables': var_names, 'coefficients': coefficients.tolist(),
                   'x_test': x_test, 'y_test': y_test, 'y_test_predict': y_test_predict, 'intercept': intercept}

    return output_dict


# This function fits a Polynomial Regression model to the input dataframe
def polynomial_model(data, prediction, choice, degree):
    var_names = list(data.columns.values)
    var_names.remove(prediction)
    x = np.array(data.drop([prediction], 1))
    y = np.array(data[prediction])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    poly_features = PolynomialFeatures(degree=degree)
    x_train_poly = poly_features.fit_transform(x_train)
    poly_model = linear_model.LinearRegression()
    poly_model.fit(x_train_poly, y_train)
    y_train_predict = poly_model.predict(x_train_poly)
    y_test_predict = poly_model.predict(poly_features.fit_transform(x_test))
    rmse_test = round(np.sqrt(mean_squared_error(y_test, y_test_predict)), 3)
    r2_test = round(r2_score(y_test, y_test_predict), 3)
    coefficients = np.around(poly_model.coef_, 3)
    output_dict = {'model': "Polynomial Model", 'test_rmse': rmse_test, 'test_r2': r2_test,
                   'Independent variables': var_names, 'coefficients': coefficients.tolist(),
                   'x_test': x_test, 'y_test': y_test, 'y_test_predict': y_test_predict,
                   'degree': degree}
    return output_dict


# This function fits a Ridge Regression model to the input dataframe
def ridgeprediction(data, prediction):
    var_names = list(data.columns.values)
    var_names.remove(prediction)
    x = np.array(data.drop([prediction], 1))
    y = np.array(data[prediction])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    rdg = Ridge(alpha=0.05)
    rdg.fit(x_train, y_train)
    rdg.score(x_train, y_train)
    y_train_predict = rdg.predict(x_train)
    y_test_predict = rdg.predict(x_test)
    test_rmse = round((np.sqrt(mean_squared_error(y_test, y_test_predict))), 3)
    test_r2 = round(r2_score(y_test, y_test_predict), 3)
    coefficients = np.around(rdg.coef_, 3)
    intercept = np.around(rdg.intercept_, decimals=3)

    output_dict = {'model': "Ridge Regression", 'test_rmse': test_rmse,
                   'test_r2': test_r2, 'Independent variables': var_names, 'coefficients': coefficients.tolist(),
                   'x_test': x_test, 'y_test': y_test, 'y_test_predict': y_test_predict, 'intercept': intercept}
    return output_dict


# This functions fits a Random Forest Regression model to the input dataframe
def RanForest(data, prediction):
    var_names = list(data.columns.values)
    var_names.remove(prediction)
    x = np.array(data.drop([prediction], 1))
    y = np.array(data[prediction])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    regressor = RandomForestRegressor()
    regressor.fit(x_train, y_train)
    y_train_predict = regressor.predict(x_train)
    y_test_predict = regressor.predict(x_test)
    test_rmse = round((np.sqrt(mean_squared_error(y_test, y_test_predict))), 3)
    test_r2 = round(r2_score(y_test, y_test_predict), 3)

    output_dict = {'model': "Random Forest Regression", 'test_rmse': test_rmse,
                   'test_r2': test_r2, 'Independent variables': var_names,
                   'x_test': x_test, 'y_test': y_test, 'y_test_predict': y_test_predict}

    return output_dict


# This function fits a Support Vector Regression to the input dataframe
def SVRegression(data, prediction):
    var_names = list(data.columns.values)
    var_names.remove(prediction)
    x = np.array(data.drop([prediction], 1))
    y = np.array(data[prediction])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    regressor = SVR(kernel='rbf')
    regressor.fit(x_train, y_train)
    y_train_predict = regressor.predict(x_train)
    y_test_predict = regressor.predict(x_test)
    test_rmse = round((np.sqrt(mean_squared_error(y_test, y_test_predict))), 3)
    test_r2 = round(r2_score(y_test, y_test_predict), 3)

    output_dict = {'model': "Support Vector Regression", 'test_rmse': test_rmse,
                   'test_r2': test_r2, 'Independent variables': var_names,
                   'x_test': x_test, 'y_test': y_test, 'y_test_predict': y_test_predict}

    return output_dict

#This function creates the corresponding plots and output messages of the calculated curve fits
def show_output(output_dict, choice, prediction):
    # Case when the user selects linear regression, polynomial regression or ridge regression
    if choice in [1, 2, 3]:
        variables = output_dict.get('Independent variables')
        coefficients = output_dict.get('coefficients')
        equation = ""

        line0 = "Regression Model: " + str(output_dict.get('model'))
        line1 = "Error: " + str(output_dict.get('test_rmse'))
        line2 = "R-squared: " + str(output_dict.get('test_r2'))
        line3 = "Variables: " + str(variables)
        line4 = "Coefficients: " + str(coefficients)

        if choice in [1,3]:
            for i in range(len(variables)):
                if i < len(variables) - 1:
                    equation += str(coefficients[i]) + "(" + str(variables[i]) + ")" + " + "
                else:
                    equation += str(coefficients[i]) + "(" + str(variables[i]) + ")"
            if output_dict.get('intercept') < 0:
                line5 = "Equation: Y = " + str(equation) + str(" ") + str(output_dict.get('intercept'))
            else:
                line5 = "Equation: Y = " + str(equation) + str("+") + str(output_dict.get('intercept'))
        else:
            line5 = ""

        win = Toplevel(bg="sky blue", bd=5)
        win.title("Modeling Results")
        win.geometry('300x170')
        labels_font = font.Font(family="Times New Roman", size=14)
        Grid.rowconfigure(win, 0, weight=1)
        Grid.rowconfigure(win, 1, weight=1)
        Grid.rowconfigure(win, 2, weight=1)
        Grid.rowconfigure(win, 3, weight=1)
        Grid.rowconfigure(win, 4, weight=1)
        Grid.rowconfigure(win, 5, weight=1)
        Grid.columnconfigure(win, 0, weight=1)

        label0 = ttk.Label(win, text=line0, font=labels_font, background="white", foreground="blue")
        label1 = ttk.Label(win, text=line1, font=labels_font, background="white")
        label2 = ttk.Label(win, text=line2, font=labels_font, background="white")
        label3 = ttk.Label(win, text=line3, font=labels_font, background="white")
        label4 = ttk.Label(win, text=line4, font=labels_font, background="white")
        label5 = ttk.Label(win, text=line5, font=labels_font, background="white")

        label0.grid(row=0, column=0, sticky=N + S + E + W)
        label1.grid(row=1, column=0, sticky=N + S + E + W)
        label2.grid(row=2, column=0, sticky=N + S + E + W)
        label3.grid(row=3, column=0, sticky=N + S + E + W)
        label4.grid(row=4, column=0, sticky=N + S + E + W)
        label5.grid(row=5, column=0, sticky=N + S + E + W)

    # Case when the user selects Random Forest or Support Vector Regression
    elif choice in [4, 5]:
        line0 = "Regression Model: " + str(output_dict.get('model'))
        line1 = "Error: " + str(output_dict.get('test_rmse'))
        line2 = "R-squared: " + str(output_dict.get('test_r2'))
        line3 = "Variables: " + str(output_dict.get('Independent variables'))

        win = Toplevel()
        win.resizable(True, True)
        win.config(bg="sky blue", bd=5)
        win.title("Modeling Results")
        win.geometry('300x170')
        labels_font = font.Font(family="Times New Roman", size=14)
        Grid.rowconfigure(win, 0, weight=1)
        Grid.rowconfigure(win, 1, weight=1)
        Grid.rowconfigure(win, 2, weight=1)
        Grid.columnconfigure(win, 0, weight=1)

        label0 = ttk.Label(win, text=line0, font=labels_font, background="white", foreground="blue")
        label1 = ttk.Label(win, text=line1, font=labels_font, background="white")
        label2 = ttk.Label(win, text=line2, font=labels_font, background="white")
        label3 = ttk.Label(win, text=line3, font=labels_font, background="white")

        label0.grid(row=0, column=0, sticky=N + S + E + W)
        label1.grid(row=1, column=0, sticky=N + S + E + W)
        label2.grid(row=2, column=0, sticky=N + S + E + W)
        label3.grid(row=3, column=0, sticky=N + S + E + W)

    var_names = output_dict.get('Independent variables')
    x_test = output_dict.get('x_test')
    y_test = output_dict.get('y_test')
    y_test_predict = output_dict.get('y_test_predict')

    # Creating the corresponding graphs of the curve fitting model

    if choice == 2:
        degree = output_dict.get('degree')
        sns.regplot(x=y_test, y=y_test_predict, color='red', scatter_kws={"s": 80}, line_kws={"color": "black"},
                    order=1,
                    ci=None)
        # Add Labels
        plt.figure(100)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediction Error Plot")
        plt.grid(True, color='#d3d3d3', linestyle='-')

        for i in range(len(var_names)):
            sns.regplot(x=x_test[:, i], y=y_test_predict, color='red', scatter_kws={"s": 80},
                        line_kws={"color": "black"}, order=degree, ci=None)
            plt.figure(i)
            # Add Labels
            plt.xlabel(str(var_names[i]))
            plt.title(prediction + "-vs-" + str(var_names[i]))
            plt.ylabel(prediction)
            plt.grid(True, color='#d3d3d3', linestyle='-')
        plt.show()

    else:

        for i in range(len(var_names)):
            plt.figure(i - 1)
            sns.regplot(x=x_test[:, i], y=y_test_predict, color='red', scatter_kws={"s": 80},
                        line_kws={"color": "black"}, ci=None)
            # Add Labels
            plt.xlabel(str(var_names[i]))
            plt.ylabel(prediction)
            plt.title(prediction + "-vs-" + str(var_names[i]))
            plt.grid(True, color='#d3d3d3', linestyle='-')

        # Add Labels
        plt.figure(100)
        sns.regplot(x=y_test, y=y_test_predict, color='red', scatter_kws={"s": 80}, line_kws={"color": "black"},
                    order=1,
                    ci=None)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediction Error Plot")
        plt.grid(True, color='#d3d3d3', linestyle='-')
        plt.show()


# This main function creates a GUI where the user can enter a csv file and the values needed to fit a curve
def main():
    # This function changes the status of the polynomial degree to fit depending on the curve option the user selects
    def disable_degree(event):
        if model.get() == 'Polynomial Regression':
            degree_entry.configure(state='normal')
        else:
            degree_entry.configure(state='disabled')

    # this function starts the process of calculation for the curve fits and present the outputs of the models
    def calculate():
        menu_dict = {"Linear Regression": 1, "Polynomial Regression": 2, "Ridge Regression": 3,
                     "Random Forest Regression": 4,
                     "Support Vector Regression": 5}
        filename = file_name.get()
        ind_vars = independent_variables.get()
        target_var = target_variable.get()
        choice = menu_dict[model.get()]
        all_columns = get_all_columns(ind_vars, target_var)
        sep = separator.get()
        data = import_data(filename, all_columns, sep)
        degree = int(degree_entry.get())

        linear_output = linearprediction(data, target_var)
        polynomial_output = polynomial_model(data, target_var, choice, degree)
        ridge_output = ridgeprediction(data, target_var)
        ranforest_output = RanForest(data, target_var)
        SVR_output = SVRegression(data, target_var)

        if choice == 1:
            show_output(linear_output, choice, target_var)
            selected = linear_output.get('test_r2')
        elif choice == 2:
            show_output(polynomial_output, choice, target_var)
            selected = polynomial_output.get('test_r2')
        elif choice == 3:
            show_output(ridge_output, choice, target_var)
            selected = ridge_output.get('test_r2')
        elif choice == 4:
            show_output(ranforest_output, choice, target_var)
            selected = ranforest_output.get('test_r2')
        else:
            show_output(SVR_output, choice, target_var)
            selected = SVR_output.get('test_r2')

        # Creating lists with all the models names and their respective accuracies
        model_names = ["Linear Regression", "Polynomial Regression", "Ridge Regression", "Random Forest Regression",
                       "Support Vector Regression"]
        r_square_list = [linear_output.get('test_r2'), polynomial_output.get('test_r2'),
                         ridge_output.get('test_r2'),
                         ranforest_output.get('test_r2'), SVR_output.get('test_r2')]
        best_model = r_square_list[0]  # Setting the first model to be the best one to enter the following loop.

        # Comparing all models R2 and selecting the best model
        i = 0
        index = 0
        while i in range(4):
            if best_model < r_square_list[i+1]:
                best_model = r_square_list[i+1]
                index = i+1
                i +=1
            else:
                i += 1

        rsquared = float(accuracy.get())
        # If the selected model has low accuracy or the selected model is not the best model, then the program
        # displays a recommendation
        if selected < rsquared and choice != index + 1:
            r = "The Recommended Model is: " + str(model_names[index])
            # Asking the user to choose to show or not the results of the recommended model
            display = control1.get()
            win2 = Toplevel()
            win2.config(bg="red", bd=5)
            win2.title("Recommended Modeling Results")
            win2.geometry('300x170')
            labels_font = font.Font(family="Times New Roman", size=14)
            Grid.rowconfigure(win2, 0, weight=1)
            Grid.columnconfigure(win2, 0, weight=1)
            label0 = ttk.Label(win2, text=r, font=labels_font, background="white", foreground='red')
            label0.grid(row=0, column=0, sticky=N + S + E + W)
            win2.resizable(True, True)

            # Displaying the recommended model if this option is selected
            if display == 1:
                if index == 0:
                    show_output(linear_output, 1, target_var)
                elif index == 1:
                    show_output(polynomial_output, 2, target_var)
                elif index == 2:
                    show_output(ridge_output, 3, target_var)
                elif index == 3:
                    show_output(ranforest_output, 4, target_var)
                elif index == 4:
                    show_output(SVR_output, 5, target_var)

    root = Tk()
    root.config(bd=8)
    root.resizable(True, True)

    #   getting the window dimensions and placing the GUI in the center
    window_width = root.winfo_screenwidth()
    window_height = root.winfo_screenheight()
    app_width = 600
    app_height = 250
    x = (window_width / 2) - (app_width / 2)
    y = (window_height /2) - (app_height / 2)
    root.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')

    Grid.rowconfigure(root, 0, weight=1)
    Grid.rowconfigure(root, 1, weight=1)
    Grid.rowconfigure(root, 2, weight=1)
    Grid.rowconfigure(root, 3, weight=1)
    Grid.rowconfigure(root, 4, weight=1)
    Grid.rowconfigure(root, 5, weight=1)
    Grid.rowconfigure(root, 6, weight=1)
    Grid.rowconfigure(root, 7, weight=1)
    Grid.rowconfigure(root, 8, weight=1)
    Grid.rowconfigure(root, 9, weight=1)
    Grid.columnconfigure(root, 0, weight=1)
    Grid.columnconfigure(root, 1, weight=1)

    root.title("Curve Fitting Toolbox for Engineering Applications")
    main_labels_font = font.Font(family="Times New Roman", size=12)

    # creating labels
    file_label = ttk.Label(root, text="File Name:", font=main_labels_font)
    separator_label= ttk.Label(root, text="Separator:", font=main_labels_font)
    target_label = ttk.Label(root, text="Target Variable:", font=main_labels_font)
    independent_label = ttk.Label(root, text="Independent Variables:", font=main_labels_font)
    model_label = ttk.Label(root, text="Model: ", font=main_labels_font)
    accuracy_label = ttk.Label(root, text="Accuracy Level (R-squared):", font=main_labels_font)
    display_label = ttk.Label(root, text="Display Recommendation Model:", font=main_labels_font)
    degree_label = ttk.Label(root, text="Degree of the Polynomial: ", font=main_labels_font)

    # placing labels on root
    file_label.grid(row=0, column=0, sticky=N + S + E + W)
    separator_label.grid(row=1, column=0, sticky=N + S + E + W)
    target_label.grid(row=2, column=0, sticky=N + S + E + W)
    independent_label.grid(row=3, column=0, sticky=N + S + E + W)
    model_label.grid(row=4, column=0, sticky=N + S + E + W)
    degree_label.grid(row=5, column=0, sticky=N + S + E + W)
    accuracy_label.grid(row=6, column=0, sticky=N + S + E + W)
    display_label.grid(row=7, column=0, sticky=N + S + E + W)

    # creating text entries
    file_name = ttk.Entry(root, font=main_labels_font)
    file_name["width"] = 40
    separator = ttk.Entry(root, font=main_labels_font)
    separator["width"] = 40
    accuracy = ttk.Entry(root, font=main_labels_font)
    file_name["width"] = 40
    independent_variables = ttk.Entry(root, font=main_labels_font)
    independent_variables["width"] = 40
    independent_variables.insert(0, "Put variables separated by commas")
    target_variable = ttk.Entry(root, font=main_labels_font)
    target_variable["width"] = 40
    degree_entry = ttk.Entry(root, font=main_labels_font)
    degree_entry["width"] = 40
    degree_entry.insert(0, "2")
    degree_entry.configure(state='disabled')

    # placing text entry
    file_name.grid(row=0, column=1, sticky=N + S + E + W)
    separator.grid(row=1, column=1, sticky=N + S + E + W)
    target_variable.grid(row=2, column=1, sticky=N + S + E + W)
    independent_variables.grid(row=3, column=1, sticky=N + S + E + W)
    accuracy.grid(row=6, column=1, sticky=N + S + E + W)
    degree_entry.grid(row=5, column=1, sticky=N + S + E + W)

    # creating combobox
    filt = StringVar()
    filt.set("Polynomial Regression")

    model = ttk.Combobox(root)
    model.state(["readonly"])
    model["values"] = ["Linear Regression", "Polynomial Regression", "Ridge Regression", "Random Forest "
                                                                                         "Regression",
                       "Support Vector Regression"]
    model.current(0)
    model["height"] = 5
    model.bind('<<ComboboxSelected>>', disable_degree)

    # placing combobox on the root
    model.grid(row=4, column=1, sticky=N + S + E + W)

    # creating radio buttons
    control1 = IntVar()
    control1.set(0)
    radio_button1 = ttk.Radiobutton(root, value=1, variable=control1, text="Yes")
    radio_button2 = ttk.Radiobutton(root, value=0, variable=control1, text="No")

    # placing radio button
    radio_button1.grid(row=7, column=1, sticky=N + S + W + E)
    radio_button2.grid(row=8, column=1, sticky=N + S + W + E)

    # creating buttons
    calculate_button = ttk.Button(root, text="Calculate", command=lambda: calculate())
    calculate_button["width"] = 25
    cancel_button = ttk.Button(root, text="Cancel", command=root.destroy)
    cancel_button["width"] = 25

    # placing buttons
    calculate_button.grid(row=9, column=0, sticky=N + S + E + W)
    cancel_button.grid(row=9, column=1, sticky=N + S + E + W)

    root.mainloop()


if __name__ == "__main__":
    main()
