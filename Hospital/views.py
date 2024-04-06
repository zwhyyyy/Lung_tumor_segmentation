# -*- coding: utf-8 -*-
from smtplib import SMTPRecipientsRefused
from django.core import mail
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
import os
import pymysql
from django.shortcuts import redirect
from django.conf import settings
from django.shortcuts import render
import random
import string
import numpy as np
import io

import base64
from PIL import Image
import json
import re
from test import set_parse
from test import main
from test import inference_single_ct
from test import zoom_in_zoom_out
from utils.visualize import draw_result_
import matplotlib.pyplot as plt
from utils.visualize import list_name
from inference.basicInformationInterface import report_
from inference.url import url
from alipay import AliPay

args = set_parse()


def generate_verification_code(length=6):
    characters = string.digits + string.ascii_uppercase
    verification_code = ''.join(random.choice(characters) for _ in range(length))
    return verification_code


def login(request):
    if request.method == "GET":
        return render(request, "login.html")
    if request.method == "POST":
        print(1)
        json_data = request.body.decode()
        dict_data = json.loads(json_data)

        conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                               charset="utf8")
        cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
        sql = "select * from login_info where email=%s"
        cursor.execute(sql, [dict_data["email"]])
        res = cursor.fetchone()
        conn.commit()
        conn.close()
        if res is None:
            return JsonResponse({"flag": 2})
        elif res["password"] == dict_data["password"]:
            response = JsonResponse({"flag": 1})  # 设置登录后跳转的目标页面
            response.set_cookie('email', dict_data["email"], max_age=24 * 60 * 60)  # 设置 Cookie 的过期时间为一年
            return response
        else:
            return JsonResponse({"flag": 2})


def register(request):
    if request.method == "GET":
        return render(request, "register.html")
    if request.method == "POST":
        json_data = request.body.decode()
        dict_data = json.loads(json_data)
        if dict_data["btn"] == 1:
            conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                                   charset="utf8")
            cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
            sql = "select * from login_info where email=%s"
            cursor.execute(sql, [dict_data["email"]])
            res = cursor.fetchone()
            sql = "select * from email_verification where email=%s"
            cursor.execute(sql, [dict_data["email"]])
            res1 = cursor.fetchone()
            conn.commit()
            conn.close()
            if res is not None:
                return JsonResponse({"flag": 3})
            if res1 is not None:
                return JsonResponse({"flag": 4})
            ver = generate_verification_code()
            try:
                mail.send_mail(
                    subject="明日方舟医疗企业欢迎您的注册",
                    message="七月七日长生殿，夜半无人私语时。悄悄告诉你，你的验证码为" + ver + ",不要告诉别人哦",
                    from_email=settings.EMAIL_HOST_USER,
                    recipient_list=[
                        dict_data['email']
                    ]
                )
            except SMTPRecipientsRefused:
                return JsonResponse({"flag": 2})
            except Exception:
                return JsonResponse({"flag": 2})
            else:
                conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                                       charset="utf8")
                cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
                sql = "insert into email_verification values (%s,%s)"
                cursor.execute(sql, [dict_data["email"], ver])
                conn.commit()
                conn.close()
                return JsonResponse({"flag": 1})
        if dict_data["btn"] == 2:
            conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                                   charset="utf8")
            cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
            sql = "select * from login_info where email=%s"
            cursor.execute(sql, [dict_data["email"]])
            res = cursor.fetchone()
            conn.commit()
            conn.close()
            if res is not None:
                return JsonResponse({"flag": 3})
            else:
                conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                                       charset="utf8")
                cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
                sql = "select * from email_verification where email=%s"
                cursor.execute(sql, [dict_data["email"]])
                res = cursor.fetchone()
                conn.commit()
                conn.close()
                if res["verification_code"] == dict_data["verification"]:
                    conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                                           charset="utf8")
                    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
                    sql = "insert into login_info(password, email) values (%s,%s)"
                    cursor.execute(sql, [ dict_data["password"], dict_data["email"]])
                    conn.commit()
                    conn.close()
                    return JsonResponse({"flag": 1})
                else:
                    return JsonResponse({"flag": 2})


def forget(request):
    if request.method == "GET":
        return render(request, "forget.html")
    if request.method == "POST":
        json_data = request.body.decode()
        dict_data = json.loads(json_data)
        if dict_data["btn"] == 1:
            conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                                   charset="utf8")
            cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
            sql = "select * from login_info where email=%s"
            cursor.execute(sql, [dict_data["email"]])
            res = cursor.fetchone()
            sql = "select * from email_verification where email=%s"
            cursor.execute(sql, [dict_data["email"]])
            res1 = cursor.fetchone()
            conn.commit()
            conn.close()
            if res is None:
                return JsonResponse({"flag": 3})
            if res1 is not None:
                return JsonResponse({"flag": 4})

            ver = generate_verification_code()
            try:
                mail.send_mail(
                    subject="智识肺癌诊疗欢迎您的注册",
                    message="七月七日长生殿，夜半无人私语时。悄悄告诉你，你的验证码为" + ver + ",不要告诉别人哦",
                    from_email=settings.EMAIL_HOST_USER,
                    recipient_list=[
                        dict_data['email']
                    ]
                )
            except SMTPRecipientsRefused:
                return JsonResponse({"flag": 2})
            except Exception:
                return JsonResponse({"flag": 2})
            else:
                conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                                       charset="utf8")
                cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
                sql = "insert into email_verification values (%s,%s)"
                cursor.execute(sql, [dict_data["email"], ver])
                conn.commit()
                conn.close()
                return JsonResponse({"flag": 1})
        if dict_data["btn"] == 2:
            conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                                   charset="utf8")
            cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
            sql = "select * from login_info where email=%s"
            cursor.execute(sql, [dict_data["email"]])
            res = cursor.fetchone()
            conn.commit()
            conn.close()
            if res is None:
                return JsonResponse({"flag": 3})
            else:
                conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                                       charset="utf8")
                cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
                sql = "select * from email_verification where email=%s"
                cursor.execute(sql, [dict_data["email"]])
                res = cursor.fetchone()
                conn.commit()
                conn.close()
                if res is None:
                    return JsonResponse({"flag": 2})
                if res["verification_code"] == dict_data["verification"]:
                    response = JsonResponse({"flag": 1})  # 设置登录后跳转的目标页面
                    response.set_cookie('email', dict_data["email"], max_age=24 * 60 * 60)  # 设置 Cookie 的过期时间为一年
                    return response


def alter(request):
    if request.method == "GET":
        return render(request, "alter.html")
    if request.method == "POST":
        email = request.COOKIES.get("email", "")
        if email != "":
            json_data = request.body.decode()
            dict_data = json.loads(json_data)
            conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                                   charset="utf8")
            cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
            sql = "update login_info set password=%s where email=%s"
            cursor.execute(sql, [dict_data["password"], email])
            conn.commit()
            conn.close()
            return JsonResponse({"flag": 1})
        else:
            return HttpResponse("404")


def user_info(request):
    if request.method == "GET":
        email = request.COOKIES.get("email", "")
        if email != "":
            conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                                   charset="utf8")
            cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
            sql = "select * from user_profile where email=%s"
            cursor.execute(sql, [email])
            res = cursor.fetchone()
            conn.commit()
            conn.close()
            name = res["username"]
            gender = res["sex"]
            birthday = res["birthday"]
            address = res["city"]
            ID = res["horoscope"]
            password = res["job"]
            phone = res["phone"]
            return render(request, "user_information.html", locals())
    if request.method == "POST":
        email = request.COOKIES.get("email", "")
        json_data = request.body.decode()
        dict_data = json.loads(json_data)
        try:
            conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                                   charset="utf8"
                                   )
            cursor = conn.cursor()
            sql = """
                UPDATE user_profile
                SET 
                    username = %s,
                    sex = %s,
                    birthday = %s,
                    phone = %s,
                    city = %s,
                    job = %s,
                    horoscope = %s
                WHERE
                    email = %s
            """
            cursor.execute(sql, [
                dict_data["name"],
                dict_data["gender"],
                dict_data["birthday"],
                dict_data["phone"],
                dict_data["address"],
                dict_data["password"],
                dict_data["ID"],
                email
            ])
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print("Error updating user profile:", str(e))
        return redirect('user_info')


def backstage(request):
    if request.method == "GET":
        email = request.COOKIES.get("email", "")
        if email != "":
            return render(request, "backstage.html", locals())
        else:
            return HttpResponse("请先登录")


def edit(request):
    if request.method == "GET":
        email = request.COOKIES.get("email", "")
        if email != "":
            return render(request, "edit.html", locals())
        else:
            return HttpResponse("请先登录")


vis_image = 0

# def upload(request):
#     global vis_image
#     if request.method == 'POST':
#         file = request.FILES['file']
#         file_path = os.path.join(settings.MEDIA_ROOT, file.name)
#         with open(file_path, 'wb') as f:
#             for chunk in request.FILES['file'].chunks():
#                 f.write(chunk)
#         file_name = "../static/media/" + file.name
#         response = HttpResponseRedirect('/handle/')  # 重定向到处理页面
#         response.set_cookie('file', file_name, max_age=24 * 60 * 60)  # 设置Cookie
#         return response
#     else:
#         return HttpResponse('404')
#
#
# def handle(request):
#     global vis_image
#     if request.method == "GET":
#         filename = request.COOKIES.get("file", "")
#         print(filename)
#         return render(request, "handle.html")
#     if request.method == "POST":
#         filename = request.COOKIES.get("file", "")
#         json_data = request.body.decode()
#         dict_data = json.loads(json_data)
#         print(filename)
#
#         if "Z" in dict_data:
#             juzhen = np.random.randint(0, 256, size=(255, 255, 255), dtype=np.uint8)   # 修改为3通道
#             image_2d = juzhen[dict_data["Z"], :, :]
#             buf = io.BytesIO()
#             img = Image.fromarray(image_2d)
#             img.save(buf, format="PNG")
#             buf.seek(0)  # 将文件指针设置到开头
#             image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#             response = HttpResponse(image_base64, content_type='text/plain')  # 修正内容类型
#             return response
#         if "X" in dict_data:
#             juzhen = np.random.randint(0, 256, size=(255, 255, 255), dtype=np.uint8)    # 修改为3通道
#             image_2d = juzhen[dict_data["X"], :, :]
#             buf = io.BytesIO()
#             img = Image.fromarray(image_2d)
#             img.save(buf, format="PNG")
#             buf.seek(0)  # 将文件指针设置到开头
#             image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#             response = HttpResponse(image_base64, content_type='text/plain')  # 修正内容类型
#             return response
#         if "Y" in dict_data:
#             juzhen = np.random.randint(0, 256, size=(255, 255, 255), dtype=np.uint8)    # 修改为3通道
#             image_2d = juzhen[dict_data["Y"], :, :]
#             buf = io.BytesIO()
#             img = Image.fromarray(image_2d)
#             img.save(buf, format="PNG")
#             buf.seek(0)  # 将文件指针设置到开头
#             image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#             response = HttpResponse(image_base64, content_type='text/plain')  # 修正内容类型
#             return response
#         if "x1" in dict_data:
#             print(dict_data["x1"])
#             print(dict_data["x2"])
#             print(dict_data["y1"])
#             print(dict_data["y2"])
#             print(dict_data["z1"])
#             print(dict_data["z2"])
#             return render(request, "handle.html", locals())
#
image_train = 0
image_zoom_out_train = 0
sg_model = 0


def upload(request):
    global vis_image
    global image_train
    global image_zoom_out_train
    global sg_model
    if request.method == 'POST':
        print(111)
        file = request.FILES['file']
        file_path = os.path.join(settings.MEDIA_ROOT, file.name)
        with open(file_path, 'wb') as f:
            for chunk in request.FILES['file'].chunks():
                f.write(chunk)
        file_name = "./static/media/" + file.name
        segvol_model, data_item, categories = main(args, file_name)
        visualization_image, image, image_zoom_out = inference_single_ct(args, segvol_model, data_item,
                                                                         categories)  # 跳入下一函数

        image_train = image
        image_zoom_out_train = image_zoom_out
        sg_model = segvol_model
        vis_image = visualization_image
        image_2d_1 = vis_image[200, :, :]
        # 显示图像
        plt.imshow(image_2d_1, cmap='gray')  # 假设是灰度图像
        plt.axis('off')  # 关闭坐标轴
        plt.savefig("./static/init1234/image_2d_1.png", bbox_inches='tight', pad_inches=0)  # 保存图像，去除空白边界
        plt.close()  # 关闭图像显示窗口

        image_2d_2 = vis_image[:, 200, :]
        plt.imshow(image_2d_2, cmap='gray')  # 假设是灰度图像
        plt.axis('off')  # 关闭坐标轴
        plt.savefig('./static/init1234/image_2d_2.png', bbox_inches='tight', pad_inches=0)  # 保存图像，去除空白边界
        plt.close()  # 关闭图像显示窗口

        image_2d_3 = vis_image[:, :, 200]
        plt.imshow(image_2d_3, cmap='gray')  # 假设是灰度图像
        plt.axis('off')  # 关闭坐标轴
        plt.savefig('./static/init1234/image_2d_3.png', bbox_inches='tight', pad_inches=0)  # 保存图像，去除空白边界
        plt.close()  # 关闭图像显示窗口

        img1_loc = "../static/init1234/image_2d_1.png"
        img2_loc = "../static/init1234/image_2d_2.png"
        img3_loc = "../static/init1234/image_2d_3.png"
        print(222)
        return render(request, 'handle.html', locals())
    else:
        return HttpResponse('404')


def handle(request):
    global vis_image
    global image_train
    global image_zoom_out_train
    global sg_model
    if request.method == "GET":
        return render(request, "handle.html")
    if request.method == "POST":
        json_data = request.body.decode()
        dict_data = json.loads(json_data)
        if "Z" in dict_data:
            juzhen = vis_image  # 修改为3通道
            image_2d = juzhen[int(dict_data["Z"] / 1.6796875), :, :]
            image_2d = image_2d.astype(np.uint8)
            img = Image.fromarray(image_2d)
            buf = io.BytesIO()
            img = Image.fromarray(image_2d)
            img.save(buf, format="PNG")
            buf.seek(0)  # 将文件指针设置到开头
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            response = HttpResponse(image_base64, content_type='text/plain')  # 修正内容类型
            return response
        if "X" in dict_data:
            juzhen = vis_image
            image_2d = juzhen[:, int(dict_data["X"] / 1.6796875), :]
            image_2d = image_2d.astype(np.uint8)
            img = Image.fromarray(image_2d)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)  # 将文件指针设置到开头
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            response = HttpResponse(image_base64, content_type='text/plain')  # 修正内容类型
            return response
        if "Y" in dict_data:
            juzhen = vis_image  # 修改为3通道
            image_2d = juzhen[:, :, int(dict_data["Y"] / 1.6796875)]
            image_2d = image_2d.astype(np.uint8)
            img = Image.fromarray(image_2d)
            buf = io.BytesIO()
            img = Image.fromarray(image_2d)
            img.save(buf, format="PNG")
            buf.seek(0)  # 将文件指针设置到开头
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            response = HttpResponse(image_base64, content_type='text/plain')  # 修正内容类型
            return response
        if "x1" in dict_data:
            if dict_data["x1"] > 0:
                print(dict_data["x1"])
                print(dict_data["x2"])
                print(dict_data["y1"])
                print(dict_data["y2"])
                print(dict_data["z1"])
                print(dict_data["z2"])
                list1 = []
                list1.append(dict_data["z1"] / 1.6796875 * 0.125)
                list1.append(dict_data["y1"] / 1.6796875)
                list1.append(dict_data["x1"] / 1.6796875)
                list1.append(dict_data["z2"] / 1.6796875 * 0.125)
                list1.append(dict_data["y2"] / 1.6796875)
                list1.append(dict_data["x2"] / 1.6796875)

                logits_labels_record = zoom_in_zoom_out(args, sg_model, image_train, image_zoom_out_train, list1)
                if args.visualize:
                    for target, values in logits_labels_record.items():
                        image, point_prompt, box_prompt, logits = values  # logits 是切割好的向量
                        draw_result_(image, logits, args.spatial_size)
                list_now = []
                for list in list_name:
                    list = '.' + list
                    list_now.append(list)
                image_list = list_now

                report = report_()
                return JsonResponse({'image_list': image_list, 'report': report})
            else:
                logits_labels_record = zoom_in_zoom_out(args, sg_model, image_train, image_zoom_out_train, ifbox=False)
                if args.visualize:
                    for target, values in logits_labels_record.items():
                        image, point_prompt, box_prompt, logits = values  # logits 是切割好的向量
                        draw_result_(image, logits, args.spatial_size)
                list_now = []
                for list in list_name:
                    list = '.' + list
                    list_now.append(list)
                image_list = list_now
                print(report_())
                report = report_()

                return JsonResponse({'image_list': image_list, 'report': report})
        return HttpResponse(status=405)


def generate_id_code(length=6):
    characters = string.digits  # 只包含数字
    verification_code = ''.join(random.choice(characters) for _ in range(length))
    return verification_code

def result(request):
    if request.method == "GET":
        image_list_str = request.GET.get('image_list')  # 获取URL中的参数值
        image_list = json.loads(image_list_str)
        report = request.GET.get('report')
        return render(request, "result.html", {'image_list': image_list, 'report': report})
    if request.method == "POST":
        json_data = request.body.decode()
        dict_data = json.loads(json_data)
        email = request.COOKIES.get("email", "")
        id_code = generate_id_code(length=8)

        conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                               charset="utf8")
        cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
        sql = "insert into history values (%s,%s,%s,%s,%s)"
        cursor.execute(sql, [id_code, dict_data["header"], email, dict_data["note"], dict_data["report"]])
        conn.commit()
        conn.close()

        for imgic in dict_data["img"]:
            conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                                   charset="utf8")
            cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
            sql = "insert into img values (%s,%s,%s)"
            cursor.execute(sql, [email, imgic, id_code])
            conn.commit()
            conn.close()
        return redirect('result')


def history(request):
    if request.method == "GET":
        email = request.COOKIES.get("email", "")
        conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                               charset="utf8")
        cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
        sql = "select * from history where email=%s"
        cursor.execute(sql, [email])
        res = cursor.fetchall()
        conn.commit()
        conn.close()

        headers = []
        numbers = []
        contents = []
        if res:
            for row in res:
                headers.append(row["header"])
                numbers.append(row["id"])
                contents.append(row["note"])
        pairs = zip(headers, numbers, contents)

        context = {
            'pairs': pairs,
        }
        return render(request, "history.html", context)


def jump(request):
    if request.method == "GET":
        email = request.COOKIES.get("email", "")
        number = request.GET.get("number")
        conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                               charset="utf8")
        cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
        sql = "select * from history where email=%s and id =%s"
        cursor.execute(sql, [email, number])
        res = cursor.fetchone()
        conn.commit()
        conn.close()
        report = res["comment"]

        image_list = []
        conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                               charset="utf8")
        cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
        sql = "select * from img where email=%s and id=%s"
        cursor.execute(sql, [email, number])
        res1 = cursor.fetchall()
        conn.commit()
        conn.close()
        if res1:
            for row in res1:
                image_list.append(row['img'])
        return JsonResponse({'image_list': image_list, "report": report})
    else:
        return HttpResponse("404")


def delete(request):
    if request.method == "GET":
        number = request.GET.get('number')
        print(number)
        conn = pymysql.connect(host="localhost", port=3306, user="root", password="123456", db="hospital",
                               charset="utf8")
        cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
        sql = "delete from history where id=%s"
        cursor.execute(sql, [number])
        sql = "delete from img where id=%s"
        cursor.execute(sql, [number])
        conn.commit()
        conn.close()
        return JsonResponse({'flag': 1})
    else:
        return HttpResponse("404")


def getanwser(query):
    res1 = re.search("类型", query)
    res2 = re.search("到什么", query)
    res3 = re.search("活", query)
    res4 = re.search("基因检测", query)
    res5 = re.search("治疗", query)
    res6 = re.search("影响", query)
    res7 = re.search("管理", query)
    pattern = "减轻|副作用"
    res8 = re.search(pattern, query)
    if res1 or res2 or res3 or res4 or res5 or res6 or res7 or res8:
        res9 = 1
    if res1 is not None:
        str1 = ("在上述诊断报告中已经告知您具体的类型，如果您想全方位的了解您可以参考以下:\n"
                "T1：肿瘤的最大径≤3cm。\n"
                "T1a：最大径≤1cm。\n"
                "T1b：最大径>1cm 但 ≤2cm。\n"
                "T1c：最大径>2cm 但 ≤3cm。\n"
                "T2：肿瘤的最大径>3cm 但 ≤5cm，或肿瘤具有其他特定条件（如侵犯脏层胸膜，或涉及主支气管且距离隆突≥2cm）。\n"
                "T2a：最大径>3cm 但 ≤4cm。\n"
                "T2b：最大径>4cm 但 ≤5cm。\n"
                "T3：肿瘤的最大径>5cm 但 ≤7cm，或肿瘤直接侵犯到特定结构（如胸壁或膈肌），或在同一肺叶内有两个或更多独立的肿瘤灶。\n"
                "T4：肿瘤的最大径>7cm，或肿瘤侵犯到更为关键的结构（如纵隔、心、大血管），或存在恶性胸膜积液，或在同一肺的不同叶内有两个或更多独立的肿瘤灶。\n")
        return str1
    if res2 is not None:
        str1 = ("在上述的诊断报告中已经描述了具体的类型，您可以根据类型进行判别，如下：\n"
                "T代表肿瘤的大小，分为T1-T4，数字越大代表肿瘤的直径越大，浸润的部位越多；\n"
                "I期（可细分为IA期和IB期）还没有扩散到淋巴结或身体远端部位。\n"
                "II期可细分为IIA期和IIB期。\n"
                "IIA期指肿瘤大于4cm，小于等于5cm，没有扩散到附近淋巴结或身体远端部位。\n"
                "IIB期指肿瘤小于等于5cm，扩散到同侧支气管或肺门淋巴结，但没有扩散到身体远端部位；或肿瘤大于5cm，小于等于7cm，没有扩散到附近淋巴结或身体远端部位。\n"
                "III期肺癌被分为IIIA、IIIB和IIIC三个亚期，这个时期的肺癌除了尚未发生远端转移外，肿瘤各状态均较II期更加严重。\n"
                "当肿瘤转移到了身体远端部位时，意味着疾病已经进展到了IV期。\n"
                "肿瘤扩散到胸腔内为IVA期，当肿瘤已经扩散到胸腔外时，则为IVB期。\n"
                )
        return str1
    if res3 is not None:
        str1 = ("作为一个诊断系统我只能为您提供一些具体的数据，具体的还是需要去医院就诊。\n"
                "早期肺癌（I期和II期）：较小的肿瘤（如直径小于3cm），未侵犯周围结构且未转移，通常属于I期或II期，5年存活率相对较高，可以达到60%到80%或更高，特别是如果通过手术完全切除。\n"
                "中期肺癌（III期）：较大的肿瘤，或者肿瘤已侵犯到周围结构或局部淋巴结，但没有远处转移，属于III期，5年存活率可能在15%到50%之间，具体取决于多种因素。\n"
                "晚期肺癌（IV期）：任何大小的肿瘤，但已有远处转移，属于IV期，5年存活率通常低于10%。\n"
                "这些数字是根据群体统计得出的大致估计，实际上每个患者的情况都是独一无二的。治疗进展，如靶向治疗和免疫治疗，已经显著改善了某些患者的预后。\n"
                "此外，具体的存活率数据可能随时间改变，随着治疗方法的改进和新技术的应用而提高。\n"
                "最准确的预后评估需要结合完整的临床评估、详细的分期信息以及可能的分子和遗传标志物分析。\n"
                "因此，对于具体个体的存活率估计，最好是与医疗保健提供者讨论，他们可以提供最适合个人情况的信息。\n"
                "还请您和您的家属保持乐观心态，您一定可以被治疗好的。\n")
        return str1
    if res4 is not None:
        str1 = ("对于基因检测，首先您应该先了解到您患病是什么类型的肺部肿瘤，再根据以下内容做判断。"
                "肿瘤是一种复杂的基因疾病，它的发展与多种基因突变密切相关。这些基因突变，通常称为驱动基因，是肿瘤细胞生长和繁殖的主要原因。不同类型的肺癌具有不同的驱动基因，这对于治疗方法的选择至关重要。通过精确识别肿瘤中的驱动基因突变，医生可以为患者选择最合适的靶向药物，从而大大提高治疗效果，延长患者的生存时间，并改善生活质量。"
                "肺腺癌是最常见的肺癌类型之一，绝大多数肺腺癌患者都可以通过基因检测来发现特定的驱动基因突变。一旦发现这些突变，就可以匹配相应的靶向药物，为患者提供个性化的治疗方案。因此，对肺腺癌患者进行常规基因检测是非常推荐的做法。"
                "与肺腺癌相比，肺鳞癌患者中发生基因突变的可能性较低，但并非不存在。这意味着通过基因检测，部分肺鳞癌患者也有可能找到适合自己的靶向治疗方案。因此，尽管基因突变的比例不高，肺鳞癌患者仍然应考虑进行基因检测。"
                "对于小细胞肺癌（SCLC）患者来说，情况则有所不同。小细胞肺癌发生基因突变的机率非常低，且目前针对这类突变的靶向药物极少，这使得基因检测在小细胞肺癌的治疗中作用有限。尽管如此，对于特定患者，进行基因检测可能有助于发现潜在的治疗机会，尽管这种情况较为罕见。"
                "具体来说您可以根据您的情况来判断需不需要做基因检测")
        return str1
    if res5 is not None:
        str1 = ("肺癌的治疗选择依赖于多个因素，包括癌症的类型，疾病的阶段，患者的整体健康状况以及个人偏好。\n"
                "以下是根据肺癌的不同阶段常见的治疗选择，包括它们可能的费用和副作用。需要注意的是，费用会因地区、医院、保险覆盖范围以及治疗计划的具体内容而有很大差异。\n"
                "早期阶段（I-II期）: \n"
                "治疗选择：手术切除、可能的辅助化疗、放射治疗（对于不能手术的患者）\n"
                "费用：手术治疗费用可能在3万元到10万元之间。辅助化疗和放射治疗可能增加数万元。(目前以纳入医保范围)\n"
                "副作用：手术风险（如感染、术后疼痛）、化疗的副作用（如恶心、脱发、疲劳）、放疗的副作用（如皮肤烧伤、疲劳）\n"
                "中期阶段（III期）\n"
                "治疗选择：化疗加放射治疗（同时化疗）、手术（某些情况下）、靶向治疗（对于具有特定遗传变异的肿瘤）、免疫治疗。\n"
                "费用：化疗加放射治疗总费用可达数万元，靶向治疗和免疫治疗的费用可能更高，每月费用可达数万元至数十万元，具体取决于药物类型。(目前以纳入医保范围)\n"
                "副作用：除上述副作用外，靶向治疗和免疫治疗可能引起肌疹、肝功能异常、免疫相关副作用等。\n"
                "晚期阶段（IV期）\n"
                "治疗选择：靶向治疗、免疫治疗、化疗、姑息治疗以减轻症状。\n"
                "费用：长期靶向治疗和免疫治疗的费用高昂，每月费用可能持续在数万元至数十万元。(目前以纳入医保范围)\n"
                "副作用：根据治疗类型而异，但管理晚期癌症的副作用和提高生活质量成为主要关注。\n"
                "")
        return str1
    if res6 is not None:
        str1 = ("早期阶段（I-II期）\n"
                "手术：手术是早期肺癌的主要治疗方式。手术后的恢复期可能会限制身体活动和日常功能几周到几个月。疼痛、疲劳和呼吸困难是常见的短期影响。\n"
                "放射治疗：如果接受放射治疗，可能会经历疲劳和特定治疗区域的皮肤变化，这些可能会影响日常活动和个人护理的能力。\n"
                "化疗：化疗可能引起的副作用，如恶心、脱发、增加感染风险，可能会对日常生活造成中断，影响饮食、社交活动和工作。\n"
                "中期阶段（III期）\n"
                "化疗加放射治疗：联合治疗的副作用可能更加显著，包括更加严重的疲劳、消化道反应和血液计数的变化，这可能要求患者需要更多的休息和调整日常活动。\n"
                "靶向治疗和免疫治疗：这些治疗可能有不同的副作用谱，包括皮疹、腹泻、肝功能异常等，可能需要定期的医学监测和药物调整，影响日程安排。\n"
                "晚期阶段（IV期）\n"
                "全身治疗：晚期肺癌通常接受靶向治疗、免疫治疗或化疗。治疗的副作用，如疲劳、消化问题和免疫相关副作用，可能成为日常生活的一个持续性影响，需要长期管理。\n"
                "姑息治疗：旨在缓解症状和改善生活质量，可能包括疼痛管理和呼吸支持，要求与医疗团队紧密合作，以调整治疗计划满足变化的需要。\n"
                "综合影响\n"
                "情绪和心理健康：在所有阶段，肺癌的诊断和治疗都可能带来情绪波动、焦虑和抑郁，影响生活质量和日常功能。\n"
                "社会和工作生活：根据治疗的影响，患者可能需要调整工作计划、休假或寻求社会支持。\n"
                "每个患者对治疗的反应都是独一无二的，与医疗团队紧密沟通，了解可能的副作用和管理策略，对于维持日常生活的质量和处理治疗带来的挑战至关重要。\n")
        return str1
    if res7 is not None:
        str1 = ("1、管理疼痛 \n"
                "药物治疗：使用医生推荐的止痛药，包括非处方药和处方药。\n"
                "非药物治疗：尝试热敷、冷敷、按摩、放松技巧或冥想等方法。\n"
                "2、缓解呼吸困难\n"
                "保持室内空气清新：使用空气净化器，避免烟雾和污染。\n"
                "氧疗：对于严重的呼吸问题，医生可能推荐使用氧气。\n"
                "3、管理消化系统问题\n"
                "饮食调整：小餐频食，避免高脂、辛辣或难以消化的食物。\n"
                "药物治疗：使用医生推荐的药物来控制恶心、呕吐或便秘。\n"
                "4、处理情绪变化\n"
                "心理支持：考虑寻求心理咨询或加入支持小组，与他人分享经历和感受。\n"
                "放松和压力管理：练习瑜伽、冥想、深呼吸或其他放松技巧。\n"
                "5、营养和水分\n"
                "保持水分：充分摄入水分，特别是在接受化疗期间。\n"
                "营养丰富的饮食：尽量吃富含蛋白质和营养的食物，以支持身体的恢复。\n")
        return str1
    if res8 is not None:
        str1 = ("1、恶心和呕吐\n"
                "药物：使用抗恶心药物，如5-HT3受体拮抗剂、NK1受体拮抗剂或其他医生推荐的药物。\n"
                "饮食调整：吃小而频繁的餐食，避免油腻、辛辣或刺激性食物。在化疗当天或之前，尝试吃些干燥、易消化的食物。\n"
                "2、管理脱发\n"
                "心理准备：了解脱发是暂时性的，头发通常在治疗结束后会重新长回。\n"
                "头皮护理：使用温和的洗发水，避免强烈化学产品或热风吹风。\n"
                "3、缓解口腔问题\n"
                "口腔卫生：定期使用温和的、不含酒精的漱口水清洁口腔。\n"
                "避免刺激性食物：避免酸性、辛辣或过硬的食物，以减少口腔不适。\n"
                "4、控制疼痛\n"
                "药物治疗：根据疼痛程度使用非处方止痛药或医生开具的处方止痛药。\n"
                "非药物方法：热敷、冷敷、按摩、放松技巧或其他互补疗法。\n"
                "5、提高心理和情绪健康\n"
                "心理支持：心理咨询、支持小组或与家人和朋友的交流可以帮助处理情绪反应。\n"
                "放松技巧：冥想、深呼吸练习和瑜伽等可以帮助减轻压力和焦虑。\n"
                "6、水分和电解质平衡\n"
                "充足水分摄入：保持充足的水分摄入，特别是在接受化疗或出现腹泻时。\n")
        return str1
    str1 = "请输入正确的提示词或者与此系统相关的问题，谢谢！\n"
    return str1


def chatgpt(request):
    if request.method == "GET":
        return render(request, "chatgpt.html")
    if request.method == "POST":
        json_data = request.body.decode()
        dict_data = json.loads(json_data)  # 使用 request.POST 获取 POST 请求的数据
        query = dict_data["value"]
        reply = getanwser(query)

        # 构建回复的 JSON 数据
        response_data = {
            'data': {
                'info': {
                    'text': reply
                }
            }
        }

        return JsonResponse(response_data)


def paycheck(request):
    if request.method == "POST":
        # GET 请求没有 body，应该使用 GET 参数
        json_data = request.body.decode()
        dict_data = json.loads(json_data)
        # 假设你已经正确配置了 alipay 的参数
        alipay = AliPay(
            appid='9021000135610129',
            app_notify_url=None,
            app_private_key_string=open(settings.ALIPAY_KEYS_DIR / "app_private_key.pem").read(),
            alipay_public_key_string=open(settings.ALIPAY_KEYS_DIR / "alipay_public_key.pem").read(),
            sign_type="RSA2",
            debug=True
        )
        # 构建支付参数
        order_string = alipay.api_alipay_trade_page_pay(
            subject='光谱行动',  ## 交易主题
            out_trade_no=str(generate_id_code(10)),  ## 订单号
            total_amount=dict_data["money"],  ## 使用前端传递的金额
            return_url="http://127.0.0.1:8000/backstage/?flag=1",  ##  请求支付，之后及时回调的一个接口
            notify_url="http://127.0.0.1:8000/backstage/?flag=1"  ##  通知地址，
        )
        url = "https://openapi-sandbox.dl.alipaydev.com/gateway.do?" + order_string
        return JsonResponse({"url": url})