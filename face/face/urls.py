"""face URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
import state.views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', state.views.home, name='home'),
    path('upload/', state.views.upload, name='upload'),
    path('picture/', state.views.picture, name='picture'),
    path('show/<int:profile_id>/', state.views.show, name='show'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)





# upload tag 형식
    # <form method='post' action="/upload/" enctype='multipart/form-data'>
    #   {% csrf_token %}
    # <div class="avatar-upload">
    #   <div class="avatar-edit">
    #     <!-- input name, action -->
        
    #       <input type='file' name = "profile" id="imageUpload" accept=".png, .jpg, .jpeg" />
    #       <label for="imageUpload"></label>

    #   </div>    
    #   <div class="avatar-preview">
    #     <div id="imagePreview" style="background-image: url(http://www.ajou.ac.kr/_resources/_prof_photo/201510389.jpg);">
    #     </div>
    #   </div>
      
    #   <br>
    #   <div class="row justify-content-center">
    #       <div class="col-6"><button type="button" class="button btn btn-primary">Upload</button></div>
    #   </div>
    #   </form>
      
    #   <p><a href="{% url 'home' %}">Return to home</a></p>  