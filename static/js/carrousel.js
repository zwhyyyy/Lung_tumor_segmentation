/**
 * Created by Haier on 2017/8/28.
 */

window.onload = function () {
    var wrap = document.getElementById("wrap");
    var slide = document.getElementById("slide");
    var lis = slide.children[0].children;
    //console.log(lis);
    var arrow = document.getElementById("arrow");
    var arrRight = document.getElementById("arrRight");
    var arrLeft = document.getElementById("arrLeft");

    //配置单  规定了每张图片的大小 层级 不透明度
    var list =[
        {
            "width": 300,
            "top": 20,
            "left": 0,
            "opacity": 0.2,
            "zIndex": 2
        },//0
        {
            "width": 500,
            "top": 70,
            "left": 50,
            "opacity": 0.8,
            "zIndex": 3
        },//1
        {
            "width": 700,
            "top": 100,
            "left": 200,
            "opacity": 1,
            "zIndex": 4
        },//2
        {
            width: 500,
            top: 70,
            left: 600,
            opacity: 0.8,
            zIndex: 3
        },//3
        {
            "width": 300,
            "top": 20,
            "left": 850,
            "opacity": 0.2,
            "zIndex": 2
        }//4
    ];

    window.onblur = function(){
        clearInterval(timer);
    }
    window.onfocus = function(){
        timer = setInterval(arrRight.onclick,1000)
    }

    var timer = null;
    wrap.onmouseover = function () {
        animate(arrow, {'opacity': 1})
        clearInterval(timer);
    };

    wrap.onmouseout = function () {
        animate(arrow, {'opacity': 0})
        //先清除定时器
        clearInterval(timer);
     timer = setInterval(arrRight.onclick,1000)
    };
    //调用配置起始位置
    config();

    //思想 当点击上一张的时候 吧list列表里中第一个配置移到最后然后在调用一次config() 重新渲染新的结果
    //数组的pop删除最后一个元素  push向数组末尾添加一元素  unshift从前面添加元素 shift 从前面删除


    //节流阀
    var off = true;

    //下一张
    arrRight.onclick = function (){
        if(off === true){
            list.unshift(list.pop());
            config();
            off = false;
        };
    };

    //上一张
    arrLeft.onclick = function(){
        if(off === true){
            list.push(list.shift());
            config();
            off = false;
        };
    };

  timer = setInterval(arrRight.onclick,1000);

    //起始位置
    function config(){
        for(var i=0;i < lis.length;i++){
            animate(lis[i],list[i],function(){
                //渲染完成后等于 true
                off = true;
            });
        };
    };
};
