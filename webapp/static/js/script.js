var x=document.getElementById('fat').offsetWidth;

function getImagePreview(event) {
  var image = URL.createObjectURL(event.target.files[0]);
  var imagediv = document.getElementById('posh');
 // var rut = document.getElementById('abc');
  var newimg = document.createElement('img');
  imagediv.innerHTML = '';
  newimg.src = image;
  newimg.width = x/2.5;
  imagediv.appendChild(newimg);
}

function deblur1() {
  var ruth = document.getElementById('abc');
  // ruth.innerHTML = '<div id="preview"><input type="file" name="upload_file" class="form-control" placeholder="Enter Name" id="upload_file" onchange="getImagePreview(event)"></div>';
  //ruth.innerHTML = '<form method="post" action="/" enctype="multipart/form-data" ><dl><p><input type="file" name="file" class="form-control" autocomplete="off" required></p></dl><p><input type="submit" value="Submit" class="btn btn-info"></p></form>'
  //ruth.innerHTML = '<div style="text-align:center;" id="posh"></div><br><form method="post" action="/" enctype="multipart/form-data" ><dl><div><p><input type="file" name="file" class="form-control" autocomplete="off" required onchange="getImagePreview(event)"></p><div></dl><p><input type="submit" value="Submit" class="btn btn-info" ></p></form>'
  ruth.innerHTML = '<div style="text-align:center;" id="posh"></div><br><form method="post" action="/" enctype="multipart/form-data" ><dl><div><p><input type="file" name="file" class="form-control" autocomplete="off" required onchange="getImagePreview(event)"></p></div></dl><br><input type="submit" value="Submit" hidden/><button class="bbb""><div style = "z-index: 100; background-color: black; height: 40px; width: 120px; border-radius: 8px;text-align:center;"><div style = "margin-top: 5px;">Submit</div></div></button><br></form>';
} 

function bgm()
      { 
        const eloe=document.getElementById("imgg21");
        eloe.innerHTML='';

        console.log(document.body.style.backgroundImage='url("/static/images/background_5.png")')
        console.log(document.body.style.transition='background-image 1.5s linear')
      }

var loader=document.getElementById("load1");
window.addEventListener("load",function(){
  loader.style.display="none";
})

function navigateToAnotherPage() {
  // Change the URL to the desired webpage
  window.location.href = "youtube.com";}


console.log(document.getElementById('logo').offsetWidth)
console.log(document.getElementById('logodiv').offsetWidth)
console.log(x)
var p=document.getElementById('logo').offsetWidth;
if(x<=463){
  var element = document.getElementById("r1");
  element.classList.remove("container");
  element.classList.remove("text-center");
  element = document.getElementById("r2");
  element.classList.remove("align-items-start");
  element.classList.remove("row");

  document.getElementById('logo').width=document.getElementById('logodiv').offsetWidth;

}
