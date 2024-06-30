import streamlit as st
from streamlit.components.v1 import html
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 80px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(0, 200, 0, 200),
        border_style="inset",
        border_width=px(0),
        color="rgb(99,140,80)" 
    )

    body = p(
        id='myFooter',
        style=styles(
            margin=px(0, 0, 0, 0),
            padding=px(5),
            font_size="0.8rem",
            color="green"
        )
    )
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

    js_code = '''
    <script>
    function rgbReverse(rgb){
        var r = rgb[0]*0.299;
        var g = rgb[1]*0.587;
        var b = rgb[2]*0.114;
        
        if ((r + g + b)/255 > 0.5){
            return "rgb(99, 140, 80)"
        }else{
            return "rgb(250, 250, 250)"
        }
        
    };
    var stApp_css = window.parent.document.querySelector("#root > div:nth-child(1) > div > div > div");
    window.onload = function () {
        var mutationObserver = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    /***********************************************/
                    var bgColor = window.getComputedStyle(stApp_css).backgroundColor.replace("rgb(", "").replace(")", "").split(", ");
                    var fontColor = rgbReverse(bgColor);
                    var pTag = window.parent.document.getElementById("myFooter");
                    pTag.style.color = fontColor;
                    /***********************************************/
                });
            });
            
            /**Element**/
            mutationObserver.observe(stApp_css, {
                attributes: true,
                characterData: true,
                childList: true,
                subtree: true,
                attributeOldValue: true,
                characterDataOldValue: true
            });
    }
    

    </script>
    '''
    html(js_code)


def footer():
    myargs = [
        "Jasmine 2024 with ❤"
    ]
    layout(*myargs)
