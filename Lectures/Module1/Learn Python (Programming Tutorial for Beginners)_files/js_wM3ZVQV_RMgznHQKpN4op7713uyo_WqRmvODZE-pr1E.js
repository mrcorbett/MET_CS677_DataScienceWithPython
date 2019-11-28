(function($) {
	Drupal.behaviors.toc = {
		attach: function (context) {

			var options = {
				tocContainer: ".title-sidebar-links",
				selectors: '.autoscroll'
			};

			var headingOffsets = [];

			var updateHeadingOffsets = function(){
				headingOffsets = [];
				$(options.selectors).each(function(i, element) {
					headingOffsets.push($(element).offset().top);
				});
			};

			$(document).ready(function() {
				
				// Check if hash is present in url
				if(window.location.hash) {
					var target = $(window.location.hash);

					// Open the accordion if relevant hash is present
					if(!target.parent().hasClass("title-accordion-open") && target.parent().hasClass("title-accordion")) {
						target.trigger("click");	
					}
				}
			});


			// $(window).on("load", function(){
			// 	updateHeadingOffsets();
			// 	$(window).scroll(highlightOnScroll);
			// });

			// var highlightOnScroll = function(e) {
			// 	var windowTop = $(window).scrollTop();
			// 	var closest = Number.MAX_VALUE;
			// 	var closestIndex = 0;
			// 	for(var i=0; i<headingOffsets.length; i++){
			// 		differenceFromTop = Math.abs(headingOffsets[i] - windowTop);
			// 		if(differenceFromTop < closest) {
			// 			closestIndex = i;
			// 			closest = differenceFromTop;
			// 		}
			// 	}

			// 	$(options.tocContainer + " " + "li").removeClass("title-link-active");
			// 	$(options.tocContainer + ' li:eq('+ closestIndex +')').addClass("title-link-active");

			// 	var hash = $(options.tocContainer + ' a:eq('+ closestIndex +')').attr('href');

			// 	if(hash.length) {
			// 		if(history.pushState) {
			// 			history.pushState(null, null, hash);
			// 		} else {
			// 			document.location.hash = hash;
			// 		}
			// 	}
			// }



			// var scrollTo = function(target){
			// 	$('html, body').animate({
			// 		scrollTop: target.offset().top
			// 	}, 500)
			// 	.promise().then(
			// 		function(){
			// 			if(!target.parent().hasClass("title-accordion-open")) {
			// 				target.trigger("click");	
			// 			}
			// 			updateHeadingOffsets();
			// 			highlightOnScroll();
			// 		}
			// 		);
			// }

			// $(document).on("click", ".toc-link", function(e){
			// 	e.preventDefault();
			// 	var target = $($(this).attr('href'));
			// 	scrollTo(target);
			// });

			// if(window.location.hash) {
			// 	var target = $(window.location.hash);
			// 	scrollTo(target);
			// }

			// $('.views-submit-button', context).css('float', 'none');
			// $('.views-exposed-widget', context).css('float', 'none');

			$(".title-accordion div", context).hide();
			
			$(document, context).delegate('.title-accordion h3', 'click', function (e) {
				e.preventDefault();
				$(this).next().slideDown(600, updateHeadingOffsets);
				$(this).parent().removeClass("title-accordion").addClass("title-accordion-open");
			});

			$(document, context).delegate('.title-accordion-open h3', 'click', function (e) {
				e.preventDefault();
				$(this).next().slideUp(1000, updateHeadingOffsets);
				$(this).parent().removeClass("title-accordion-open").addClass("title-accordion");

			});
		}
	};

})(jQuery);;
