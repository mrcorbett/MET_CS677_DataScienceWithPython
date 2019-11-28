// Avoid `console` errors in browsers that lack a console.x
//
//(function($) {
//
//(function() {
//    var method;
//    var noop = function () {};
//    var methods = [
//        'assert', 'clear', 'count', 'debug', 'dir', 'dirxml', 'error',
//        'exception', 'group', 'groupCollapsed', 'groupEnd', 'info', 'log',
//        'markTimeline', 'profile', 'profileEnd', 'table', 'time', 'timeEnd',
//        'timeline', 'timelineEnd', 'timeStamp', 'trace', 'warn'
//    ];
//    var length = methods.length;
//    var console = (window.console = window.console || {});
//
//    while (length--) {
//        method = methods[length];
//
//        // Only stub undefined methods.
//        if (!console[method]) {
//            console[method] = noop;
//        }
//    }
//}());
//
//})(jQuery);


// Place any jQuery/helper plugins in here.
;
(function($) {
  $(document).ready(function() {
    if (getCookie("privacy-policy")) {
      $(".privacy-policy").hide();
      $(".adblock-code").css("margin-bottom", "0");
    } else {
      $(".privacy-policy").show();
    }

    // Load Get-App url
    var os;
    os = getOS();

    var getApp_playStoreUrl =
        "https://play.google.com/store/apps/details?id=com.programiz.learnpython&referrer=utm_source%3Dprogramiz-website%26utm_medium%3Dandroid%26utm_campaign%3Dget-app",
      getApp_appStoreUrl = "https://apps.apple.com/app/apple-store/id1472188189?pt=120228772&ct=get-app-ios&mt=8",
      getApp_desktopUrl = "/learn-python?ref=get-app";

    var getAppButton = $(".nav-get-python-app");

    if (os == "Android") {
      getAppButton.attr("href", getApp_playStoreUrl);
    } else if ((os == "iOS")) {
      getAppButton.attr("href", getApp_appStoreUrl);
    } else {
      getAppButton.attr("href", getApp_desktopUrl);
    }

    if ($(".sticky-sidebar-title").length !== 0) {
      var headerTop = $(".header-wrapper").height();
      var footerTop = $("#footer").offset().top;
      var windowTop;

      $(window).scroll(function() {
        windowTop = $(window).scrollTop();
        if (windowTop > headerTop + 20) {
          $(".sticky-sidebar-title").css({
            position: "fixed",
            bottom: "inherit",
            top: "0"
          });
          if (windowTop + 600 > footerTop - 75) {
            var footerScroll = windowTop + 600 - footerTop - 500;
            $(".sticky-sidebar-title").css({
              top: "inherit",
              bottom: $("#footer").height() + footerScroll
            });
          }
        } else {
          $(".sticky-sidebar-title").css({ position: "inherit" });
        }
      });
    }

    //------------------- When search icon is click ---------------

    $(".search-icon").click(function(e) {
      e.preventDefault();
      $("#search-area").fadeIn(400);

      var headerHeight = $("#programiz-header").height();
      var searchFormHeight = $("#search-area form").height();
      var marginTop = (headerHeight - searchFormHeight) / 2;

      $("#search-area").css({
        position: "fixed",
        width: "100%",
        background: "#DDD !important",
        top: "0",
        "z-index": "100",
        height: "110px"
      });

      $("#search-area form").css({
        margin: "0 auto",
        "margin-top": "40px"
      });

      $("#search-area .gsc-input input").focus();
    });

    $("#search-area span").click(function(e) {
      e.preventDefault();
      $("#search-area").hide();
      $("#search-area").css("position", "static");
    });

    // ---------------------- Premium Fancy Bar ----------------------------------------------

    function setCookie(key, value, expireDate) {
      var expires = new Date();
      expires.setDate(expires.getDate() + expireDate);
      document.cookie =
        key + "=" + value + ";path=/;expires=" + expires.toUTCString();
    }

    function getCookie(key) {
      var re = new RegExp(key + "=([^;]+)");
      var value = re.exec(document.cookie);
      var keyValue = value != null ? unescape(value[1]) : null;
      return keyValue === null ? false : true;
    }

    // var crossButtonClicked = false;
    // var cross = getCookie("programizFancyBar");

    // if(cross != null) {
    //     crossButtonClicked = true;
    // }

    // $("footer").on("click" ,".fancy-cross", function(e){
    //     e.preventDefault;
    //     crossButtonClicked = true;

    //     $(".fancy-premium").slideUp(200);
    //     setCookie("programizFancyBar", true);
    // })

    // -----------------------  sidebar Menu ------------------------------------------------

    var relativeURL = window.location.pathname;
    // $("#programiz-left-sidebar a[href= '" + relativeURL + "']").addClass("currentPage");

    // $(".submenu a").click(function() {

    //     $("topmenu ").find(".currentPage").removeClass("currentPage");
    //     $(this).parent().addClass("currentPage");
    // });

    $(".header-list-menu a").each(function() {
      var href = $(this).attr("href");

      if (href.indexOf(relativeURL) > -1) {
        $(this)
          .closest("li")
          .addClass("currentPage");
        return false;
      }
    });

    $(".header-list-menu .submenu")
      .not(":has(.currentPage)")
      .addClass("visuallyhidden");
    $(".header-list-menu .submenu:has(.currentPage)").addClass(
      "visuallynothidden"
    );

    $(".header-list-menu .visuallyhidden")
      .prev()
      .append('<span class="indicator_hide"></span>');
    $(".header-list-menu .visuallynothidden")
      .prev()
      .append('<span class="indicator_show"></span>');

    $(".header-list-menu .topmenu h3").click(function() {
      var all = $(".header-list-menu.visuallynothidden");
      all.removeClass("visuallynothidden");
      all.addClass("visuallyhidden");
      $(this)
        .next()
        .toggleClass("visuallyhidden visuallynothidden");

      $(".indicator_hide").remove();
      $(".indicator_show").remove();
      $(".header-list-menu .visuallyhidden")
        .prev()
        .append('<span class="indicator_hide"></span>');
      $(".header-list-menu .visuallynothidden")
        .prev()
        .append('<span class="indicator_show"></span>');
    });

    // ------------------------------ Screen size (after changed as well) ----------------------

    var width = $(window).width();
    var document_width = $(window).width();

    if (width >= 1120) {
      // var recommendedReadingHeight = $(newFunction()).height() + 3;
      // $(".tutorial-toc").height(recommendedReadingHeight);

      if ($(".tutorial-toc").length !== 0) {
        $(window).scroll(function() {
          var footerHeight = $("#footer").outerHeight();
          var documentHeight = $(document).height();
          var windowTop = $(window).scrollTop();
          var recommendedReadingHeight = $(".tutorial-toc").height();
          var remainingHeight =
            documentHeight -
            recommendedReadingHeight -
            footerHeight -
            15 -
            15 -
            40;

          if (windowTop > remainingHeight) {
            var top = windowTop - remainingHeight;
            $(".tutorial-toc").css("top", -top);
          } else {
            $(".tutorial-toc").css("top", 0);
          }
        });
      }
    } else {
      $(".tutorial-toc-label").on("click", function(e) {
        e.preventDefault();
        closeTutorialToc();
      });

      var tocNotClickFlag = true;

      var position = $(window).scrollTop();
      var scrollTimer = null;

      $(".tutorial-toc a").on("click", function(e) {
        e.preventDefault();

        $("body").css("overflow", "initial");
        $("#programiz-main-content").css({ opacity: 1 });
        $(".tutorial-toc > ul").hide();
        $(".toc-background-click").removeClass("toc-background-click");
        $(".tutorial-toc-label").html("Contents");

        var goToPosition = $(this).attr("href");

        $("html, body")
          .animate(
            { scrollTop: $(goToPosition).offset().top },
            "slow",
            function() {}
          )
          .promise()
          .then(function() {
            if (tocNotClickFlag) {
              tocNotClickFlag = false;
            }
            $(".tutorial-toc-label").fadeIn(100);
          });
      });

      if (tocNotClickFlag) {
        $(window).scroll(function() {
          if (scrollTimer) {
            clearTimeout(scrollTimer);
          }
          scrollTimer = setTimeout(handleScroll, 25);
        });
      }

      function handleScroll() {
        var scroll = $(window).scrollTop();
        if (scroll > position) {
          if ($(".tutorial-toc-label").is(":visible")) {
            if (tocNotClickFlag) {
              $(".tutorial-toc-label").fadeOut(100);
            }
          }
        } else {
          if ($(".tutorial-toc-label").is(":hidden")) {
            $(".tutorial-toc-label").fadeIn(100);
          }
        }
        tocNotClickFlag = true;
        position = scroll;
      }

      $("body").on("click", ".toc-background-click", function(e) {
        e.preventDefault();
        closeTutorialToc();
        $("body").css("overflow", "initial");
      });
    }

    function closeTutorialToc() {
      if ($(".tutorial-toc > ul").is(":visible")) {
        $(".tutorial-toc > ul").hide();

        $(".toc-background-click").remove();
        $(".tutorial-toc-label").html("Contents");
        $("body").css("overflow", "initial");
      } else {
        $(".tutorial-toc > ul").show();
        $("body").append("<div class='toc-background-click'></div>");
        $(".tutorial-toc-label").html(
          'Close <span class="material-icons toc-label-close">close</span>'
        );
        $("body").css("overflow", "hidden");
      }
    }

    if (width >= 1120 && $(".tutorial-toc > ul").is(":hidden")) {
      $(".tutorial-toc > ul").show();
      $(".toc-background-click").removeClass("toc-background-click");
    }

    changeWidth(width);

    $(window).resize(function() {
      width = $(window).width();

      if ($(this).width() != width) {
        width = $(this).width();
      }

      changeWidth(width);
    });

    function loadMobileNavigation() {
      if ($(".nav-mobile").length === 0) {
        $("body").prepend(
          '<nav class="nav-mobile"><div><a class="material-icons back-nav">arrow_back</a><a class="material-icons close-nav">clear</a></div><ul class="nav-mobile-topmenu"></ul></nav>'
        );
      }

      var navMobile = $(".nav-mobile");

      // Setup close button click event
      var closeButton = $(".nav-mobile .material-icons.close-nav");
      closeButton.click(function(e) {
        e.preventDefault();

        $("html").css("overflow", "initial");

        if ($(window).scrollTop() <= 71) {
          $(".main-nav-wrapper").removeClass("sticky");
        }

        navMobile.remove();
        loadMobileNavigation();
      });

      // Setup back button click event
      var backButton = $(".nav-mobile .material-icons.back-nav");
      backButton.click(function(e) {
        e.preventDefault();

        $(".nav-mobile div h2").remove();
        $(".nav-mobile div").prepend($(".header-list-menu h2").clone());

        $(".nav-mobile-outer-topmenu > a").show();
        selectedMenu.animate(
          {
            left:
              parseInt(selectedMenu.css("left"), 10) == 0
                ? -$(selectedMenu).outerWidth() - $(window).width()
                : 0
          },
          500
        );
        backButton.hide();
        closeButton.show();
      });

      // Setup menu items
      if ($(".nav-mobile-topmenu li").length == 0) {
        if ($(".header-list-menu")[0] && $(".header-list-menu .topmenu")[0]) {
          var headerElement = $(".header-list-menu .currentPage")
            .parent()
            .prev()
            .text();

          $(".nav-mobile div").prepend("<h2>" + headerElement + "</h2>");

          var menuItems = $(".header-list-menu .topmenu")
            .children()
            .each(function(index) {
              // Set h3 titles as first level items
              var topLevelItem = $(this)
                .find("h3")
                .text();

              var prevTopLevelItem = $(this).prev()[0]
                ? $(this)
                    .prev()
                    .find("h3")
                    .text()
                : null;

              var nextTopLevelItem = $(this).next()[0]
                ? $(this)
                    .next()
                    .find("h3")
                    .text()
                : null;

              var subMenu = $(this)
                .find(".submenu")
                .clone();

              var currentTopLevel = $(
                '<li class="nav-mobile-outer-topmenu"><a class="' +
                  (subMenu.hasClass("visuallynothidden")
                    ? "nav-mobile-visuallynothidden"
                    : "") +
                  '" href="#">' +
                  topLevelItem +
                  '<span class="material-icons nav-right">navigate_next</span></a></li>'
              ).appendTo(".nav-mobile-topmenu");
              currentTopLevel.append(subMenu);

              // Add top level elements to submenus
              // if (prevTopLevelItem != null) {
              //   var prevTopLevel = $(
              //     '<li><ul><li class="nav-mobile-prevmenu"><a href="#">' +
              //       prevTopLevelItem +
              //       '<span class="material-icons nav-right">navigate_next</span></a></li></ul></li>'
              //   ).prependTo(subMenu);
              // }
              // if (nextTopLevelItem != null) {
              //   var nextTopLevel = $(
              //     '<li><ul><li class="nav-mobile-nextmenu"><a href="#">' +
              //       nextTopLevelItem +
              //       '<span class="material-icons nav-right">navigate_next</span></a></li></ul></li>'
              //   ).appendTo(subMenu);
              // }
            });

          // Set inner menu items to far left.
          $(".nav-mobile-topmenu .submenu").each(function(index) {
            $(this).css("left", -$(this).outerWidth() - $(window).width());
          });

          // highlight currentpage parent

          if ($(".submenu li").hasClass("currentPage")) {
            $(".currentPage")
              .parent()
              .prev()
              .addClass("currentPageParent");
          } else {
            $(".nav-mobile > div .back-nav").hide();
            $(".nav-mobile div").prepend("<h2>Recommended Readings</h2>");
            $(".nav-mobile-topmenu").append(
              $(".recommended-links .item-list ul li").clone()
            );

            $(".nav-mobile-topmenu li").show();
          }
        } else if (
          $(".header-list-menu")[0] &&
          !$(".header-list-menu .topmenu")[0]
        ) {
          $(".nav-mobile > div .back-nav").hide();
          $(".nav-mobile").addClass("nav-mobile-no-submenu");
          $(".nav-mobile div").prepend($(".header-list-menu h2").clone());
          $(".nav-mobile-topmenu").append($(".header-list-menu ul li").clone());
          $(".nav-mobile-topmenu").addClass("no-submenu-mobile");
          $(".nav-mobile-topmenu li").show();
        } else {
          // For non-article menus
          $(".nav-mobile-topmenu").append(
            $(".main-nav ul li:not(.mobile-menu):not(.desktop-only)").clone()
          );
          $(".nav-mobile-topmenu li").show();
        }
      }

      // Link Click handler
      $(".nav-mobile-topmenu li a").on("click", function(e) {
        // Top level menu items click handler.

        if ($(this).attr("href") == "#") {
          e.preventDefault();

          // $(".currentPageParent").removeClass("currentPageParent");
          // $(this).addClass("currentPageParent");

          $(".nav-mobile div h2").remove();
          var headerElement1 = $(".header-list-menu .currentPage")
            .parent()
            .prev()
            .text();

          var clickedText = $(this).text();

          clickedText = clickedText.replace("navigate_next", "");

          if (clickedText != headerElement1) {
            headerElement1 = clickedText;
          }

          $(".nav-mobile div").prepend("<h2>" + headerElement1 + "</h2>");

          var parentMenu = $(this).parent();
          if (
            parentMenu.hasClass("nav-mobile-prevmenu") ||
            parentMenu.hasClass("nav-mobile-nextmenu")
          ) {
            var menuToBeClosed = $(this).closest(".submenu");
            // menuToBeClosed.animate(
            //   {
            //     left:
            //       parseInt(menuToBeClosed.css("left"), 10) == 0
            //         ? -$(menuToBeClosed).outerWidth() - $(window).width()
            //         : 0
            //   },
            //   12333
            // );

            menuToBeClosed.css(
              "left",
              parseInt(menuToBeClosed.css("left"), 10) == 0
                ? -$(menuToBeClosed).outerWidth() - $(window).width()
                : 0
            );

            $(".nav-mobile-outer-topmenu > a").hide();

            selectedMenu = parentMenu.hasClass("nav-mobile-prevmenu")
              ? $(this)
                  .closest(".nav-mobile-outer-topmenu")
                  .prev()
                  .find(".submenu")
              : $(this)
                  .closest(".nav-mobile-outer-topmenu")
                  .next()
                  .find(".submenu");
          } else {
            selectedMenu = $(this).next();
          }
          selectedMenu.show();
          setTimeout(
            () =>
              selectedMenu.animate(
                {
                  left:
                    parseInt(selectedMenu.css("left"), 10) == 0
                      ? -$(selectedMenu).outerWidth() - $(window).width()
                      : 0
                },
                500
              ),
            0
          );

          // menuToBeClosed.show();

          // closeButton.hide();
          backButton.show();
        } else {
          // Normal menu click handler
          closeButton.trigger("click");
        }
      });

      // Hide the menu and back button on first load
      // backButton.hide();
      // navMobile.hide();

      // Replicate click for current page
      $(".mobile-menu > a.material-icons").on("click", function(event) {
        $(".nav-mobile .nav-mobile-topmenu .currentPage a")
          .closest("ul.visuallynothidden")
          .prev()
          .trigger("click");
        $(".nav-mobile-outer-topmenu > a").hide();
      });
    }

    // Border-bottom on Main nav

    var pathName = window.location.pathname;
    var matchedUrl = "";
    var matchedElement;

    $(".main-nav a").each(function() {
      var link = $(this).attr("href");

      if (link != "") {
        var substringExists = checkSubstring(link);

        if (substringExists === true) {
          if (link.length > matchedUrl.length) {
            matchedUrl = link;
          }
        }
      }
    });

    $(".main-nav a[href='" + matchedUrl + "']").addClass("current-main-nav");

    function checkSubstring(link) {
      if (pathName.indexOf(link) !== -1) {
        return true;
      } else {
        return false;
      }
    }

    function changeWidth(width) {
      //  $(window).off('scroll');

      //--------------------------------- width less than 900 --------------------------------------
      if (width < 900) {
        $(window).scroll(function() {
          // height of main-nav-wrapper is 57
          if ($(window).scrollTop() > 64) {
            $(".main-nav-wrapper").addClass("sticky");
            $(
              "#main-wrapper-two-column, #main-wrapper-three-column, #title-page-wrapper"
            ).css("margin-top", "55px");
          } else {
            $(".main-nav-wrapper").removeClass("sticky");
            $(
              "#main-wrapper-two-column, #main-wrapper-three-column, #title-page-wrapper"
            ).css("margin-top", "0px");
          }
        });

        // $(".main-nav li").removeClass("current-nav-li");

        var classFoundInUrl = false;
        var relativeURL = window.location.pathname;

        $(".main-nav ul a").each(function() {
          var className = $(this)
            .attr("class")
            .split(" ")[0];
          if (relativeURL.indexOf(className) > -1) {
            $(this)
              .parent()
              .addClass("current-nav-li");
            classFoundInUrl = true;
            return false;
          }
        });

        if (classFoundInUrl == false)
          $(".main-nav .home-icon")
            .parent()
            .addClass("current-nav-li");

        // navReset();

        //------------------ Hide Menubar and  Show menuicon for width < 900 ---------------------------

        // $("#programiz-left-sidebar").hide();

        $(".main-nav .current-nav-li")
          .parent()
          .children()
          .hide();
        $(".main-nav .current-nav-li").show();
        // ------------------- Update this code if new theme subtheme of Ultimate is created ------------------------------//
        if (
          relativeURL.indexOf("/c-programming") == 0 ||
          relativeURL.indexOf("/cpp-programming") == 0 ||
          relativeURL.indexOf("/java-programming") == 0 ||
          relativeURL.indexOf("/kotlin-programming") == 0 ||
          relativeURL.indexOf("/python-programming") == 0
        ) {
          $(".main-nav .current-nav-li")
            .next()
            .show();
        }
        $(".main-nav .home-icon")
          .parent()
          .show();
        $(".main-nav").show();

        if ($(".nav-mobile").length === 0) {
          loadMobileNavigation();
        }
        // $(".mobile-menu").show();

        var navRecommendedVisible = false;
        var navRecommended = $(".recommended-links h3 span");

        $(".recommended-links h3").on("click", function(e) {
          e.preventDefault();

          if (!navRecommendedVisible) {
            $(".recommended-links ul").show();
            navRecommendedVisible = true;
            navRecommended.html("expand_less");
            return;
          } else {
            $(".recommended-links ul").hide();
            navRecommendedVisible = false;
            navRecommended.html("expand_more");
            return;
          }
        });
      } else {
        if (width < 1120) {
          // $(".tutorial-toc-label").on("click", function(e) {
          //   e.preventDefault();
          //   if ($(".tutorial-toc > ul").is(":visible")) {
          //     $(".tutorial-toc > ul").hide();
          //     $("#programiz-main-content").css({ opacity: 1 });
          //     $(".tutorial-toc-label").html("Contents");
          //   } else {
          //     $(".tutorial-toc > ul").show();
          //     $("#programiz-main-content").css({ opacity: 0.2 });
          //     $(".tutorial-toc-label").html(
          //       'Close <span class="material-icons toc-label-close">close</span>'
          //     );
          //   }
          // });

          // $(".tutorial-toc a").on("click", function(e) {
          //   $(".tutorial-toc > ul").hide();
          //   $("#programiz-main-content").css({ opacity: 1 });
          //   $(".tutorial-toc-label").html("Contents");
          // });

          if (
            $(".main-nav-wrapper").hasClass("menu-three-column") &&
            $(".nav-mobile").length === 0
          ) {
            loadMobileNavigation();
          }
        }

        if (width >= 1120 && $(".tutorial-toc > ul").is(":hidden")) {
          $(".tutorial-toc > ul").show();
        }

        // $("#programiz-left-sidebar").show();

        $(".main-nav .current-nav-li")
          .parent()
          .children()
          .show();
        // $(".mobile-menu").hide();

        var longList = false;
        $(".large-list-sidebar td").each(function(i) {
          if (i > 10) {
            $(this).addClass("visuallyhidden");
          }
          if (i == 11) {
            longList = true;
          }
        });

        $(".large-list-sidebar li").each(function(i) {
          if (i > 10) {
            $(this).addClass("visuallyhidden");
          }
          if (i == 11) {
            longList = true;
          }
        });

        if (longList === true) {
          if ($(".view-more-button .material-icons").length == 0) {
            $(".large-list-sidebar").append(
              "<div class='view-more-button'><i class='material-icons'>expand_more</i></div>"
            );
          }
        }

        $(".view-more-button").on("click", function() {
          $(".large-list-sidebar td").removeClass("visuallyhidden");
          $(".view-more-button").hide();

          $(".large-list-sidebar li").removeClass("visuallyhidden");
          $(".view-more-button").hide();
        });

        // $(".recommended-links ul").show();
        $(".recommended-links h3 span").hide();

        if ($(".recommended-links").length !== 0) {
          $(window).scroll(function() {
            var footerHeight = $("#footer").outerHeight();
            var documentHeight = $(document).height();
            var windowTop = $(window).scrollTop();
            var recommendedReadingHeight = $(".recommended-links").height();
            var remainingHeight =
              documentHeight -
              recommendedReadingHeight -
              footerHeight -
              15 -
              15 -
              30;

            if (windowTop > remainingHeight) {
              var top = windowTop - remainingHeight;
              $(".recommended-links").css("top", -top);
            }
          });
        }
      }
    }

    //-------------------- Reset elements realated to naviation when screen is changed ----------------------
    function navReset() {
      //$("#search-area").hide();
      $(".main-nav-wrapper").css({
        position: "static"
      });
      $("#programiz-header").css({
        position: "static"
      });
      $("#main-wrapper").css({
        "margin-top": "0"
      });
      $("#front-page-main-wrapper").css({
        "margin-top": "0"
      });
    }

    // ------------------------ Mobile Menu --------------------------------------------

    target = {};

    $(".main-nav").on("click", ".mobile-menu > a.material-icons", function(e) {
      e.preventDefault();

      if ($(window).scrollTop() <= 71) {
        $(".main-nav-wrapper").addClass("sticky");
      }

      $(".tutorial-toc > ul").hide();
      $(".toc-background-click").removeClass("toc-background-click");
      $(".tutorial-toc-label").html("Contents");

      $(".nav-mobile").show();
      $("html").css("overflow", "hidden");
    });

    // -------------------------- scroll to top -----------------------------

    var isAdBlockActive = null;

    var xhrImAds = new XMLHttpRequest();
    xhrImAds.open("GET", "/imads.js");
    xhrImAds.setRequestHeader("Access-Control-Allow-Origin", "*");
    xhrImAds.onload = function() {
      isAdBlockActive = false;
      console.log("imads-done");
    };
    xhrImAds.onerror = function() {
      isAdBlockActive = true;
      console.log("imads-fail");
      showAdBlockCreative();
    };
    xhrImAds.send();

    // $.getScript("/ads.js")
    //   .done(function(data, textStatus) {
    //     isAdBlockActive = false;
    //     console.log('ads-done');
    //   })
    //   .fail(function(jqxhr, settings, exception) {
    //     isAdBlockActive = true;
    //     showAdBlockCreative();
    //     console.log('ads-fail');
    //   });
    // $.getScript("/imads.js")
    //   .done(function(data, textStatus) {
    //     isAdBlockActive = false;
    //     console.log('imads-done');
    //   })
    //   .fail(function(jqxhr, settings, exception) {
    //     isAdBlockActive = true;
    //     showAdBlockCreative();
    //     console.log('imads-fail');
    //   });

    function showAdBlockCreative() {
      $(".adblock-creative").show();
    }

    var documentHeight = $(document).height();

    $(window).scroll(function() {
      if (
        $(this).scrollTop() > 200 &&
        $(".fancy-premium:first").is(":hidden") &&
        crossButtonClicked == false
      ) {
        $(".fancy-premium").slideDown(400);
      }

      if (
        $(this).scrollTop() <= 200 &&
        $(".fancy-premium:first").is(":visible")
      ) {
        $(".fancy-premium").slideUp(300);
      }

      if ($(this).scrollTop()) {
        $("#back-to-top").fadeIn();
      } else {
        $("#back-to-top").fadeOut();
      }

      if (
        isAdBlockActive &&
        $(this).scrollTop() / documentHeight > 0.5 &&
        !getCookie("adblock-code")
      ) {
        if ($(".adblock-code").length !== 0) {
          $(".adblock-code").slideDown(700);
        }
      }
    });

    $(".adblock-code .material-icons").click(function() {
      $(".adblock-code").hide();
      setCookie("adblock-code", false, 1);
    });

    $(".privacy-continue").click(function() {
      $(".privacy-policy").hide();
      $(".adblock-code").css("margin-bottom", 0);
      setCookie("privacy-policy", false, 364);
    });

    if ($(".privacy-policy").is(":visible")) {
      $(".adblock-code").css("margin-bottom", $(".privacy-policy").height());
    } else {
      $(".adblock-code").css("margin-bottom", "0");
    }

    // ------------------ Share button and fixed sidebar -------------------------
    $(window).on("load", function() {
      if ($(".tutorial-toc").length > 0) {
        var sidebarScrollTop = $(".tutorial-toc").offset().top;

        // $(".header-list-menu .topmenu h3").click(function() {
        //   sidebarScrollTop = $(".tutorial-toc").offset().top;
        // });

        // $(".view-more-button").click(function() {
        //   sidebarScrollTop = $(".tutorial-toc").offset().top;
        // });
      }

      if (window.location.hash) {
        $(".tutorial-toc").addClass("fix-sidebar");
      }

      if (sidebarScrollTop != null) {
        if ($(document).scrollTop() >= sidebarScrollTop) {
          $(".tutorial-toc").addClass("fix-sidebar");
        } else {
          $(".tutorial-toc").removeClass("fix-sidebar");
        }

        $(document).scroll(function() {
          if (sidebarScrollTop != null) {
            if ($(document).scrollTop() >= sidebarScrollTop) {
              $(".tutorial-toc").addClass("fix-sidebar");
            } else {
              $(".tutorial-toc").removeClass("fix-sidebar");
            }
          }
        });
      }
    });

    // Programming IDE code begin
    var ideFilter = { ideFor: "", isIdeRecommended: "" };

    function selectActiveIDEFilter(ideFilter) {
      switch (ideFilter["ideFor"]) {
        case "ide-learn":
          $(".ide-filter-learning").addClass("ide-filter-active");
          $(".ide-filter-learning")
            .siblings()
            .removeClass("ide-filter-active");
          break;

        case "ide-development":
          $(".ide-filter-development")
            .siblings()
            .removeClass("ide-filter-active");
          $(".ide-filter-development").addClass("ide-filter-active");
          break;

        default:
          $(".ide-filter-learning").removeClass("ide-filter-active");
          $(".ide-filter-development").removeClass("ide-filter-active");
      }

      if (ideFilter["isIdeRecommended"] == "ide-recommended-yes") {
        $(".ide-programiz-recommended").addClass("ide-filter-active");
      } else {
        $(".ide-programiz-recommended").removeClass("ide-filter-active");
      }
    }
    function hideAllIdeItem() {
      if ($(".ide-item").length != 0) {
        $(".ide-item").hide();
      }
    }
    function showFilteredIDEs(ideFilter) {
      if (
        ideFilter["ideFor"] == "ide-learn" &&
        ideFilter["isIdeRecommended"] == "ide-recommended-yes"
      ) {
        hideAllIdeItem();
        $(".ide-learn.ide-recommended-yes").show();
      }

      if (
        ideFilter["ideFor"] == "ide-development" &&
        ideFilter["isIdeRecommended"] == "ide-recommended-yes"
      ) {
        hideAllIdeItem();
        $(".ide-development.ide-recommended-yes").show();
      }

      if (
        ideFilter["ideFor"] == "ide-learn" &&
        ideFilter["isIdeRecommended"] == ""
      ) {
        hideAllIdeItem();
        $(".ide-learn").show();
      }

      if (
        ideFilter["ideFor"] == "ide-development" &&
        ideFilter["isIdeRecommended"] == ""
      ) {
        hideAllIdeItem();
        $(".ide-development").show();
      }

      if (
        ideFilter["ideFor"] == "" &&
        ideFilter["isIdeRecommended"] == "ide-recommended-yes"
      ) {
        hideAllIdeItem();
        $(".ide-recommended-yes").show();
      }

      if (ideFilter["ideFor"] == "" && ideFilter["isIdeRecommended"] == "") {
        $(".ide-item").show();
      }
    }

    $(function() {
      $(".dsa-examples").tabs();
    });

    function updateIdeFilterJSON(attr, value) {
      ideFilter[attr] = ideFilter[attr] == value ? "" : value;
      selectActiveIDEFilter(ideFilter);
      showFilteredIDEs(ideFilter);
    }

    $(".ide-filter-learning").click(function(e) {
      e.preventDefault();
      updateIdeFilterJSON("ideFor", "ide-learn");
    });

    $(".ide-filter-development").click(function(e) {
      e.preventDefault();
      updateIdeFilterJSON("ideFor", "ide-development");
    });

    $(".ide-programiz-recommended").click(function(e) {
      e.preventDefault();
      updateIdeFilterJSON("isIdeRecommended", "ide-recommended-yes");
    });

    var tocElements = "";
    var tocLink;
    var id;

    $(".tutorial-toc a").each(function() {
      tocLink = $(this).attr("href");
      tocElements += tocLink + ",";
    });

    tocElements = tocElements.slice(0, -1);

    $(window).scroll(function() {
      var position = window.pageYOffset;

      $(tocElements).each(function() {
        var target = $(this).offset().top;

        id = $(this).attr("id");

        var navLinks = $(".tutorial-toc li a");
        if (position + 5 >= target) {
          navLinks.removeClass("current-toc");
          $('.tutorial-toc li a[href="#' + id + '"]').addClass("current-toc");
        }
      });
    });
  });

  // Detect os
  function getOS() {
    var userAgent = window.navigator.userAgent,
      platform = window.navigator.platform,
      macosPlatforms = ["Macintosh", "MacIntel", "MacPPC", "Mac68K"],
      windowsPlatforms = ["Win32", "Win64", "Windows", "WinCE"],
      iosPlatforms = ["iPhone", "iPad", "iPod"],
      os = null;

    if (macosPlatforms.indexOf(platform) !== -1) {
      os = "Mac OS";
    } else if (iosPlatforms.indexOf(platform) !== -1) {
      os = "iOS";
    } else if (windowsPlatforms.indexOf(platform) !== -1) {
      os = "Windows";
    } else if (/Android/.test(userAgent)) {
      os = "Android";
    } else if (!os && /Linux/.test(platform)) {
      os = "Linux";
    }

    return os;
  }
})(jQuery);
;
