from django.urls import path
from . import views
urlpatterns=[
    path('',views.loginPage, name="loginPage"),
    
    path('dashboard/',views.dashboard, name="dashboard"),
    
    path('register/',views.registerPage, name="registerPage"),
    path('logout/',views.logoutUser, name="logout"),

    path('home/',views.home, name="home"),
    path('profile/',views.profile, name="profile"),
    # path('home_section/',views.home_section, name="home_section"),
    path('attendancereport/',views.attendancereport, name="attendancereport"),
     path('add_photos/', views.add_photos, name="add_photos"),
     path('facetraining/', views.facetraining, name="facetraining"),
     path('markattendance/', views.markattendance, name="markattendance"),
    #  path('data/', views.data, name="data"),
]
    
    
