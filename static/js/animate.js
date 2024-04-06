function animate(obj, json, fn) {
    clearInterval(obj.timer);
    obj.timer = setInterval(function () {
        var flag = true;
        for (var k in json) {
            if (k === "opacity") {
                var leader = getStyle(obj, k) * 100; //不能是小数  且要先把传入的透明度先乘100
                var target = json[k] * 100;
                var step = (target - leader) / 10;
                step = step > 0 ? Math.ceil(step) : Math.floor(step);
                leader = leader + step;
                obj.style[k] = leader / 100;
            } else if (k === "zIndex") {
                obj.style.zIndex = json[k];
            } else {
                var leader = parseInt(getStyle(obj, k)) || 0;
                var target = json[k];
                var step = (target - leader) / 10;
                step = step > 0 ? Math.ceil(step) : Math.floor(step);
                leader = leader + step;
                obj.style[k] = leader + "px";
            }
            if (leader != target) { //如果有一项没有到达目标 则不会清除定时器
                flag = false;
            }
        }
        if (flag) {  //直到所有的都到达目标了 才会到这里面来 清除定时器\
            clearInterval(obj.timer);
            if (fn) {
                fn();   //如果有则调用  没有则不调用 不用写
            }
        }
    }, 15);
};

function getStyle(obj,attr){
    if(obj.currentStyle){ //兼容 i 低版本浏览器
        return obj.currentStyle[attr];
    }else{
        //getComputedStyle(元素，伪元素)[属性];  伪元素没有 写null；
        return window.getComputedStyle(obj,null)[attr];  //标准浏览器
    };
};