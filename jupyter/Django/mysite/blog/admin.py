from django.contrib import admin
from .models import Post
# Register your models here.

# 관리자 페이지에서 만든 모델을 보기위해서는 모델을 사이트에 등록해야한다.
admin.site.register(Post)

# admin페이지에 로그인하기 위해서는 superuser를 생성해야한다.
