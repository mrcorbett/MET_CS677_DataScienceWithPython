// 06-2019 ggates //

var adW = 728,
    adH = 90,
    imgs = [];

var Ad = {
    getIA: function(iaName) {
        return myFT.instantAds[iaName];
    },
    byId: function(id) {
        return document.getElementById(id);
    },
    loadImages: function(imgs) {
        var tmpCnt = 0;
        for (var i = 0; i < imgs.length; i++) {
            var newImg = new Image();
            newImg.src = imgs[i].ia;

            newImg.addEventListener('load', function() {
                tmpCnt++;
                if (tmpCnt == imgs.length) {
                    Ad.init();
                }
            });
            newImg.addEventListener('error', function(e) {
                e.target.src = "images/blank.png";
            });
        }
    },
    pushImages: function() {
        imgs.push({
            ia: Ad.getIA("bg_img"),
            divid: "bg"
        }, {
            ia: Ad.getIA("doctor_img"),
            divid: "doc_image"
        }, {
            ia: Ad.getIA("cta1_img"),
            divid: "cta1"
        }, {
            ia: Ad.getIA("cta2_img"),
            divid: "cta2"
        });

        Ad.loadImages(imgs);
    },
    removeElement: function(elementId) {
        // Removes an element from the document.
        var element = document.getElementById(elementId);
        element.parentNode.removeChild(element);
    },
    xySplitter: function xySplitter(values) {
        values = values.split(",");
        return values;
    },
    xyUpdater: function(iaName, targetDiv) {
        if (Ad.getIA(iaName) !== "") {
            var coord = Ad.xySplitter(Ad.getIA(iaName));
            targetDiv.classList.remove("center");
            targetDiv.classList.remove(concept);
            targetDiv.classList.remove("defaultWidth");
            targetDiv.style.left = coord[0] + "px";
            targetDiv.style.top = coord[1] + "px";
        }
    },
    init: function() {
        myFT.applyClickTag(header_click, 1, myFT.instantAds.clickTag1_url);
        myFT.applyClickTag(footer_click, 1, myFT.instantAds.clickTag2_url);

        //apply imgs to divs
        for (i=0; i < imgs.length; i++) {
            console.log("imgs running");
            var newImg = document.createElement("img");
            newImg.src = imgs[i]["ia"];
            Ad.byId(imgs[i]["divid"]).appendChild(newImg);
        }

        //apply text to divs
        Ad.byId("doc").innerHTML = Ad.getIA("doctorInfo_txt");
        Ad.byId("address").innerHTML = Ad.getIA("address_txt");
        Ad.byId("phone").innerHTML = Ad.getIA("phone_txt");
        Ad.byId("description").innerHTML = Ad.getIA("description_txt");

        // update XY coordinates
        Ad.byId("doc_image").style.top = Ad.xySplitter(Ad.getIA("doctor_img_xy"))[1] + "px";
        Ad.byId("doc_image").style.left = Ad.xySplitter(Ad.getIA("doctor_img_xy"))[0] + "px";
        Ad.byId("content_text").style.top = Ad.xySplitter(Ad.getIA("contentBlock_txt_xy"))[1] + "px";
        Ad.byId("content_text").style.left = Ad.xySplitter(Ad.getIA("contentBlock_txt_xy"))[0] + "px";
        Ad.byId("cta1").style.top = Ad.xySplitter(Ad.getIA("cta1_img_xy"))[1] + "px";
        Ad.byId("cta1").style.left = Ad.xySplitter(Ad.getIA("cta1_img_xy"))[0] + "px";
        Ad.byId("cta2").style.top = Ad.xySplitter(Ad.getIA("cta2_img_xy"))[1] + "px";
        Ad.byId("cta2").style.left = Ad.xySplitter(Ad.getIA("cta2_img_xy"))[0] + "px";

        //FADE IN AD
        setTimeout(function(){
            container.classList.add("fadeIn");
        },500);
    }
};

myFT.on("instantads", Ad.pushImages);
