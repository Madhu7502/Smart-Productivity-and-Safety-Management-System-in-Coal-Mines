from django.contrib import admin
from django.urls import path
from application import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),

    # ---------------- Home ----------------
    path('', views.home, name='home'),

    # ---------------- Authentication ----------------
    path('admin-login/', views.admin_login, name='admin_login'),
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # ---------------- Dashboards ----------------
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('user-dashboard/', views.user_dashboard, name='user_dashboard'),

    # =================================================
    # ============== SECURITY (Admin) =================
    # =================================================
    path('admin/security/', views.security_dashboard, name='security_dashboard'),
    path('admin/security/upload/', views.security_upload, name='security_upload'),
    path('admin/security/preprocess/', views.security_preprocess, name='security_preprocess'),
    path('admin/security/split/', views.security_split, name='security_split'),
    path('admin/security/existing-model/', views.security_existing_model, name='security_existing_model'),
    path('admin/security/proposed-model/', views.security_proposed_model, name='security_proposed_model'),
    path('admin/security/performance/', views.security_performance, name='security_performance'),

    # =================================================
    # ============ PRODUCTION (Admin) =================
    # =================================================
    path('admin/production/', views.production_dashboard, name='production_dashboard'),
    path('admin/production/upload/', views.production_upload, name='production_upload'),
    path('admin/production/preprocess/', views.production_preprocess, name='production_preprocess'),
    path('admin/production/split/', views.production_split, name='production_split'),
    path('admin/production/existing-model/', views.production_existing_model, name='production_existing_model'),
    path('admin/production/proposed-model/', views.production_proposed_model, name='production_proposed_model'),
    path('admin/production/performance/', views.production_performance, name='production_performance'),

    # =================================================
    # ================= USER PREDICTIONS ==============
    # =================================================
    path('user/safety/', views.user_verify_safety, name='user_verify_safety'),
    path('user/production/', views.user_verify_production, name='user_verify_production'),
]

# Serve static files during development
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
