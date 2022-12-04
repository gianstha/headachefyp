// let btnLogin = document.getElementById("login");
// let btnSignUp = document.getElementById("signup");

// let signIn = document.querySelector(".signin");
// let signUp = document.querySelector(".signup");

// btnLogin.onclick = function(){
//     signIn.classList.add("active");
//     signUp.classList.add("inActive");
// }

// btnSignUp.onclick = function(){
//    signIn.classList.remove("active");
//    signUp.classList.remove("inActive");
// }



TweenMax.set($("#header2"), {scale:0.7});
TweenMax.set($("#header3"), {scale:0.8});
TweenMax.set($("#header"), {backgroundColor:"rgba(255,255,255,1)"});
TweenMax.set($("#circle1"),{scale:0});

TweenMax.to($("#header"), 1, {scale:1,yoyo:true,repeat:-1,opacity:1}); 
TweenMax.to($("#header"), 5, {backgroundColor:"rgba(3,255,129,1)",yoyo:true, repeat:-1}); 

TweenMax.to($("#header2"), 1, {rotation:360,repeat:-1,opacity:1});
TweenMax.to($("#header2"), 2, {scale:1,yoyo:true,repeat:-1,});
TweenMax.to($("#header3"), 2, {rotation:720,repeat:-1,ease: Power0.easeNone, y: 0 });
TweenMax.to($("#header3"), 1, {scale:0.9,yoyo:true,repeat:-1,});
TweenMax.to($("#header3"), 2, {borderRadius:100,yoyo:true,repeat:-1,});
//TweenMax.to($("#header3"),2,{scale:0,delay:10},10);
//TweenMax.to($("#header2"),2,{scale:0,delay:10},10);
//TweenMax.to($("#header"),3,{scale:3,delay:11},10);
//TweenMax.to($("#header"),3,{width:200,x:-50,delay:11},10);

//TweenMax.to($("#home"),1,{display:block});
//TweenMax.to($("#cercle1"), 2, {scale:10});
// TweenMax.set($(".desinks"), {x:-1000,color:"rgba(255,255,255,1)"});
// TweenMax.set($(".petit"), {x:1000,color:"rgba(255,255,255,1)"});

// TweenMax.to($(".desinks"), 5, {color:"rgba(3,255,129,1)",yoyo:true, repeat:-1});
// TweenMax.to($(".petit"), 5, {color:"rgba(3,255,129,1)",yoyo:true, repeat:-1});
// TweenMax.to($(".desinks"), 1, {delay:7,x:0,ease: Back.easeOut.config(0.7)});
// TweenMax.to($(".petit"), 1, {delay:7.5,x:0,ease: Back.easeOut.config(0.7)});
Back.easeOut.config(0.7);