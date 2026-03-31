from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.core.files.storage import default_storage

import os, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

# =================================================
# Globals (single-user demo mode)
# =================================================
helmet_df = None
coal_df = None

Xs_train = Xs_test = ys_train = ys_test = None
Xp_train = Xp_test = yp_train = yp_test = None

scaler_safety = None
scaler_prod = None
country_le = None

MODEL_DIR = "model"
SEC_PLOT_DIR = "static/security"
PROD_PLOT_DIR = "static/production"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SEC_PLOT_DIR, exist_ok=True)
os.makedirs(PROD_PLOT_DIR, exist_ok=True)

# =================================================
# Home
# =================================================
def home(request):
    return render(request, "home.html")

# =================================================
# Authentication
# =================================================
def admin_login(request):
    if request.method == 'POST':
        uname = request.POST['username']
        pwd = request.POST['password']
        if uname == 'admin' and pwd == 'admin':
            request.session['is_admin'] = True
            return redirect('admin_dashboard')
        messages.error(request, "Invalid admin credentials")
    return render(request, 'adminlogin.html')


def register(request):
    if request.method == 'POST':
        name = request.POST['name']
        email = request.POST['email']
        password = request.POST['password']
        confirm = request.POST['confirm_password']

        if password != confirm:
            messages.error(request, "Passwords do not match")
            return redirect('register')

        if User.objects.filter(username=email).exists():
            messages.error(request, "User already exists")
            return redirect('register')

        user = User.objects.create_user(
            username=email,
            email=email,
            password=password,
            first_name=name
        )
        user.save()
        messages.success(request, "Registration successful. Please login.")
        return redirect('login')

    return render(request, 'register.html')


def login_view(request):
    if request.method == "POST":
        uname = request.POST['username']
        pwd = request.POST['password']

        if uname == 'admin' and pwd == 'admin':
            request.session['is_admin'] = True
            return redirect('admin_dashboard')

        user = authenticate(username=uname, password=pwd)
        if user:
            login(request, user)
            return redirect('user_dashboard')

        messages.error(request, "Invalid credentials")

    return render(request, 'login.html')


def logout_view(request):
    logout(request)
    request.session.flush()
    return redirect('home')

# =================================================
# Dashboards
# =================================================
def admin_dashboard(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    return render(request, 'admin_dashboard.html')


@login_required
def user_dashboard(request):
    return render(request, 'user_dashboard.html')

# =================================================
# ========== SECURITY MODULE (Admin) ===============
# =================================================
def security_dashboard(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    return render(request, 'security_dashboard.html')


def security_upload(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    global helmet_df

    if request.method == "POST" and request.FILES.get("file"):
        path = default_storage.save("helmet.csv", request.FILES["file"])
        helmet_df = pd.read_csv(default_storage.path(path))
        default_storage.delete(path)

        table = helmet_df.head(10).to_html(classes="table table-striped", index=False)
        return render(request, "security_dashboard.html", {'table': table})

    return redirect('security_dashboard')


def security_preprocess(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    global helmet_df

    helmet_df.fillna(helmet_df.mean(numeric_only=True), inplace=True)

    le = LabelEncoder()
    for col in helmet_df.select_dtypes(include='object'):
        helmet_df[col] = le.fit_transform(helmet_df[col])

    plt.figure(figsize=(5,4))
    sns.countplot(x="Target", data=helmet_df)
    plt.title("Safe vs Hazardous")
    plot_path = f"{SEC_PLOT_DIR}/class_dist.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    table = helmet_df.head(10).to_html(classes="table table-bordered", index=False)

    return render(request, "security_dashboard.html", {
        'table': table,
        'plot': f"/{plot_path}"
    })


def security_split(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    global helmet_df, Xs_train, Xs_test, ys_train, ys_test, scaler_safety

    X = helmet_df.iloc[:, 2:16]
    y = helmet_df.iloc[:, -1]

    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_safety = StandardScaler()
    Xs_train = scaler_safety.fit_transform(Xs_train)
    Xs_test = scaler_safety.transform(Xs_test)
    joblib.dump(scaler_safety, f"{MODEL_DIR}/scaler_safety.pkl")

    return render(request, "security_dashboard.html", {
        'split': True,
        'train_size': Xs_train.shape[0],
        'test_size': Xs_test.shape[0]
    })


def security_existing_model(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')

    model = GaussianNB()
    model.fit(Xs_train, ys_train)
    joblib.dump(model, f"{MODEL_DIR}/security_nb.pkl")

    y_pred = model.predict(Xs_test)

    acc = accuracy_score(ys_test, y_pred)*100
    prec = precision_score(ys_test, y_pred, average='macro')*100
    rec = recall_score(ys_test, y_pred, average='macro')*100
    f1 = f1_score(ys_test, y_pred, average='macro')*100

    cm = confusion_matrix(ys_test, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues",
                xticklabels=['Safe','Hazardous'],
                yticklabels=['Safe','Hazardous'])
    plt.title("Naive Bayes Confusion Matrix")
    plot_path = f"{SEC_PLOT_DIR}/nb_cm.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return render(request, "security_dashboard.html", {
        'metrics': {
            'accuracy': round(acc,2),
            'precision': round(prec,2),
            'recall': round(rec,2),
            'fscore': round(f1,2),
        },
        'plot': f"/{plot_path}"
    })


def security_proposed_model(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(Xs_train, ys_train)
    joblib.dump(model, f"{MODEL_DIR}/security_xgb.pkl")

    y_pred = model.predict(Xs_test)

    acc = accuracy_score(ys_test, y_pred)*100
    prec = precision_score(ys_test, y_pred, average='macro')*100
    rec = recall_score(ys_test, y_pred, average='macro')*100
    f1 = f1_score(ys_test, y_pred, average='macro')*100

    cm = confusion_matrix(ys_test, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues",
                xticklabels=['Safe','Hazardous'],
                yticklabels=['Safe','Hazardous'])
    plt.title("XGBoost Confusion Matrix")
    plot_path = f"{SEC_PLOT_DIR}/xgb_cm.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return render(request, "security_dashboard.html", {
        'metrics': {
            'accuracy': round(acc,2),
            'precision': round(prec,2),
            'recall': round(rec,2),
            'fscore': round(f1,2),
        },
        'plot': f"/{plot_path}"
    })


def security_performance(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')

    nb = joblib.load(f"{MODEL_DIR}/security_nb.pkl")
    xgb = joblib.load(f"{MODEL_DIR}/security_xgb.pkl")

    y_nb = nb.predict(Xs_test)
    y_xgb = xgb.predict(Xs_test)

    acc = [accuracy_score(ys_test, y_nb)*100, accuracy_score(ys_test, y_xgb)*100]
    prec = [precision_score(ys_test, y_nb, average='macro')*100,
            precision_score(ys_test, y_xgb, average='macro')*100]
    rec = [recall_score(ys_test, y_nb, average='macro')*100,
           recall_score(ys_test, y_xgb, average='macro')*100]
    f1 = [f1_score(ys_test, y_nb, average='macro')*100,
          f1_score(ys_test, y_xgb, average='macro')*100]

    values = np.array([acc, prec, rec, f1]).T
    metrics = ['Accuracy','Precision','Recall','F1']
    x = np.arange(len(metrics))
    bw = 0.3

    plt.figure(figsize=(8,5))
    plt.bar(x, values[0], bw, label='NaiveBayes')
    plt.bar(x+bw, values[1], bw, label='XGBoost')
    plt.xticks(x+bw/2, metrics)
    plt.title("Security Model Comparison")
    plt.legend()
    plt.tight_layout()
    plot_path = f"{SEC_PLOT_DIR}/compare.png"
    plt.savefig(plot_path)
    plt.close()

    return render(request, "security_dashboard.html", {
        'plot': f"/{plot_path}"
    })

# =================================================
# ======= PRODUCTION MODULE (Admin) ================
# =================================================
def production_dashboard(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    return render(request, 'production_dashboard.html')


def production_upload(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    global coal_df

    if request.method == "POST" and request.FILES.get("file"):
        path = default_storage.save("coal.csv", request.FILES["file"])
        coal_df = pd.read_csv(default_storage.path(path), sep=';')
        default_storage.delete(path)

        table = coal_df.head(10).to_html(classes="table table-striped", index=False)
        return render(request, "production_dashboard.html", {'table': table})

    return redirect('production_dashboard')


def production_preprocess(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    global coal_df, country_le

    coal_df.fillna(coal_df.mean(numeric_only=True), inplace=True)

    country_le = LabelEncoder()
    coal_df['Country'] = country_le.fit_transform(coal_df['Country'])
    joblib.dump(country_le, f"{MODEL_DIR}/country_encoder.pkl")

    yearly = coal_df.groupby('Year')['Value (Million Tonnes)'].sum().reset_index()
    countrywise = coal_df.groupby('Country')['Value (Million Tonnes)'].sum().reset_index()
    countrywise = countrywise.sort_values(by='Value (Million Tonnes)', ascending=False).head(10)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(yearly['Year'], yearly['Value (Million Tonnes)'], marker='o')
    plt.title('Year vs Production')
    plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(countrywise['Country'], countrywise['Value (Million Tonnes)'], marker='o')
    plt.title('Top Countries vs Production')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)

    plot_path = f"{PROD_PLOT_DIR}/year_country.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    table = coal_df.head(10).to_html(classes="table table-bordered", index=False)

    return render(request, "production_dashboard.html", {
        'table': table,
        'plot': f"/{plot_path}"
    })


def production_split(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')
    global coal_df, Xp_train, Xp_test, yp_train, yp_test, scaler_prod

    X = coal_df.iloc[:, 0:5]
    y = coal_df.iloc[:, -1]

    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_prod = StandardScaler()
    Xp_train = scaler_prod.fit_transform(Xp_train)
    Xp_test = scaler_prod.transform(Xp_test)
    joblib.dump(scaler_prod, f"{MODEL_DIR}/scaler_prod.pkl")

    return render(request, "production_dashboard.html", {
        'split': True,
        'train_size': Xp_train.shape[0],
        'test_size': Xp_test.shape[0]
    })


def production_existing_model(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')

    model = LinearRegression()
    model.fit(Xp_train, yp_train)
    joblib.dump(model, f"{MODEL_DIR}/prod_lr.pkl")

    y_pred = model.predict(Xp_test)

    mse = mean_squared_error(yp_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yp_test, y_pred)
    r2 = r2_score(yp_test, y_pred)*100

    plt.figure(figsize=(5,5))
    plt.scatter(yp_test, y_pred, alpha=0.6)
    minv, maxv = min(yp_test.min(), y_pred.min()), max(yp_test.max(), y_pred.max())
    plt.plot([minv, maxv], [minv, maxv], '--')
    plt.title("Linear Regression: Actual vs Predicted")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.grid(alpha=0.3)
    plot_path = f"{PROD_PLOT_DIR}/lr_scatter.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return render(request, "production_dashboard.html", {
        'metrics': {
            'mse': round(mse,2),
            'rmse': round(rmse,2),
            'mae': round(mae,2),
            'r2': round(r2,2)
        },
        'plot': f"/{plot_path}"
    })


def production_proposed_model(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')

    model = DecisionTreeRegressor(random_state=42)
    model.fit(Xp_train, yp_train)
    joblib.dump(model, f"{MODEL_DIR}/prod_dt.pkl")

    y_pred = model.predict(Xp_test)

    mse = mean_squared_error(yp_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yp_test, y_pred)
    r2 = r2_score(yp_test, y_pred)*100

    plt.figure(figsize=(5,5))
    plt.scatter(yp_test, y_pred, alpha=0.6)
    minv, maxv = min(yp_test.min(), y_pred.min()), max(yp_test.max(), y_pred.max())
    plt.plot([minv, maxv], [minv, maxv], '--')
    plt.title("Decision Tree: Actual vs Predicted")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.grid(alpha=0.3)
    plot_path = f"{PROD_PLOT_DIR}/dt_scatter.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return render(request, "production_dashboard.html", {
        'metrics': {
            'mse': round(mse,2),
            'rmse': round(rmse,2),
            'mae': round(mae,2),
            'r2': round(r2,2)
        },
        'plot': f"/{plot_path}"
    })


def production_performance(request):
    if not request.session.get('is_admin'):
        return redirect('admin_login')

    lr = joblib.load(f"{MODEL_DIR}/prod_lr.pkl")
    dt = joblib.load(f"{MODEL_DIR}/prod_dt.pkl")

    y_lr = lr.predict(Xp_test)
    y_dt = dt.predict(Xp_test)

    mse = [mean_squared_error(yp_test, y_lr), mean_squared_error(yp_test, y_dt)]
    rmse = [np.sqrt(mse[0]), np.sqrt(mse[1])]
    mae = [mean_absolute_error(yp_test, y_lr), mean_absolute_error(yp_test, y_dt)]
    r2 = [r2_score(yp_test, y_lr)*100, r2_score(yp_test, y_dt)*100]

    values = np.array([mse, rmse, mae, r2]).T
    metrics = ['MSE','RMSE','MAE','R2']
    x = np.arange(len(metrics))
    bw = 0.3

    plt.figure(figsize=(8,5))
    plt.bar(x, values[0], bw, label='LinearReg')
    plt.bar(x+bw, values[1], bw, label='DecisionTree')
    plt.xticks(x+bw/2, metrics)
    plt.title("Production Model Comparison")
    plt.legend()
    plt.tight_layout()
    plot_path = f"{PROD_PLOT_DIR}/compare.png"
    plt.savefig(plot_path)
    plt.close()

    return render(request, "production_dashboard.html", {
        'plot': f"/{plot_path}"
    })

# =================================================
# ============== USER PREDICTIONS ==================
# =================================================
@login_required
def user_verify_safety(request):
    if request.method == "POST":
        fields = [float(request.POST[f]) for f in [
            'Temp_C','Humidity_Percent','CH4_ppm','H2S_ppm',
            'CO_ppm','VOC_ppm','Pressure_hPa','Mud_Level_cm',
            'Mud_Viscosity_Index','Mud_Salinity_uS_cm',
            'Accel_Z_g','Light_lux','Impact_N','Fluid_Level_cm'
        ]]

        input_df = pd.DataFrame([fields], columns=[
            'Temp_C','Humidity_Percent','CH4_ppm','H2S_ppm','CO_ppm',
            'VOC_ppm','Pressure_hPa','Mud_Level_cm',
            'Mud_Viscosity_Index','Mud_Salinity_uS_cm',
            'Accel_Z_g','Light_lux','Impact_N','Fluid_Level_cm'
        ])

        scaler = joblib.load(f"{MODEL_DIR}/scaler_safety.pkl")
        model = joblib.load(f"{MODEL_DIR}/security_xgb.pkl")

        X_scaled = scaler.transform(input_df)
        pred = model.predict(X_scaled)[0]
        result = "Hazardous" if pred == 1 else "Safe"

        return render(request, 'user_safety.html', {'prediction': result})

    return render(request, 'user_safety.html')


@login_required
def user_verify_production(request):
    if request.method == "POST":
        Year = int(request.POST['Year'])
        Country = request.POST['Country']
        G1 = float(request.POST['Growth_2021'])
        G2 = float(request.POST['Growth_2011_21'])
        Share = float(request.POST['Share_2021'])

        le = joblib.load(f"{MODEL_DIR}/country_encoder.pkl")
        Country_enc = le.transform([Country])[0]

        input_df = pd.DataFrame([[Year, Country_enc, G1, G2, Share]], columns=[
            'Year','Country',
            'Growth rate per annum 2021',
            'Growth rate per annum 2011-21','Share 2021'
        ])

        scaler = joblib.load(f"{MODEL_DIR}/scaler_prod.pkl")
        model = joblib.load(f"{MODEL_DIR}/prod_dt.pkl")

        X_scaled = scaler.transform(input_df)
        pred = model.predict(X_scaled)[0]

        return render(request, 'user_production.html',
                      {'prediction': round(pred, 2)})

    return render(request, 'user_production.html')
