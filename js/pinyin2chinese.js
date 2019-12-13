var Pinyin2Chinese = (function() {
    var Pinyin2Chinese = function(element) {
        this.curHanzi='';
        this.element = element;
    };
    Pinyin2Chinese.prototype.html_entity_decode = function(str) {
        var tarea=document.createElement('textarea');
        tarea.innerHTML = str;
        return tarea.value;
    };
    Pinyin2Chinese.prototype.setValue = function(val) {
        this.curHanzi = val;
    };

    Pinyin2Chinese.prototype.getValue = function() {
        return this.curHanzi;
    };

    Pinyin2Chinese.prototype.doGetCaretPosition = function (ctrl) {
        var CaretPos = 0;
        // IE Support
        if(document.selection) {
            ctrl.focus ();
            var Sel = document.selection.createRange ();
            Sel.moveStart ('character', -ctrl.value.length);
            CaretPos = Sel.text.length;
        } else if (ctrl.selectionStart || ctrl.selectionStart == '0') {
            // Firefox support
            CaretPos = ctrl.selectionStart;
        }
        return (CaretPos);
    };

    Pinyin2Chinese.prototype.setCaretPosition = function(ctrl, pos) {
        if(ctrl.setSelectionRange) {
            ctrl.focus();
            ctrl.setSelectionRange(pos,pos);
        } else if (ctrl.createTextRange) {
            var range = ctrl.createTextRange();
            range.collapse(true);
            range.moveEnd('character', pos);
            range.moveStart('character', pos);
            range.select();
        }
    }

    Pinyin2Chinese.prototype.update = function(pinyin) {
        pinyin = pinyin.split(/\s+/);
        var hanzi = "";
        var num = 1;
        var strSel = "";
        var strSel2 = "";
        for (i in pinyin) {
            pchar = (pinyin[i]);
            if(pchar) {
                pchar_idx = pchar.replace(/([\.\,\;\!\"\'\?\:\\A-Za-z]+)([0-9]*)/,'$2');
                pchar_val = pchar.replace(/([\.\,\;\!\"\'\?\:\\A-Za-z]+)([0-9]*)/,'$1');

                c = M[pchar_val];
                if(c) {
                    var chnlist=c.split(" ");
                    for(j in chnlist) {
                        chnitem = chnlist[j];
                        hanzi += num.toString()+ "" + chnitem.replace(/([^;]+;)/, '$1')+" ";
                        if (pchar_idx==num) {
                            strSel =chnitem.replace(/([^;]+;)/, '$1');
                            strSel2 = this.html_entity_decode(strSel);
                        }
                        num +=1;
                    }
                } else {
                    hanzi +=  pinyin[i] ;
                }
            }
        }
        this.setValue(strSel2);
        if(strSel.length>0)
            hanzi = hanzi.replace(strSel, "<font style='color:blue; background-color:yellow;'>"+strSel+"<\/font>");
        //hanzi_element = document.getElementById("hanzi");
        //hanzi_element.readonly = false;
        //hanzi_element.innerHTML = hanzi;
        //hanzi_element.readonly = true;
        if(hanzi.length>0) {
            tooltip.show(hanzi);
        } else {
            tooltip.hide();
        }
    };
    Pinyin2Chinese.prototype.getPinyin = function(input) {
        var pinyin = this.getPinyin0(input);
        if(pinyin[0].length==0 || pinyin[0].match(/^\d+$/))
            pinyin = this.getBiaodian(input);
        return pinyin;
    };
    Pinyin2Chinese.prototype.getPinyin0 = function(input) {
        var element = this.element;//input;

        var strValue = element.value;
        var selPos = this.doGetCaretPosition(element);
        var strPinyinNow = '';
        var type = 0;
        var startPos = selPos;
        var endPos = selPos;
        for(var i=selPos-1; i>=0; i--) {
            var newtype = -1;
            var curChar = strValue.substr(i,1);
            if(curChar.match(/[A-Za-z]/))
                newtype = 2;
            else if(curChar.match(/[0-9]/))
                newtype = 1;
            if(newtype<type)
                break;
            type = newtype;
            startPos = i;
            strPinyinNow = curChar + strPinyinNow;
        }
        type = 0;
        if(strPinyinNow.length>0) {
            endPos = selPos-1;
            var curChar = strPinyinNow.substr(strPinyinNow.length-1,1);
            if(curChar.match(/[A-Za-z]/))
                type = 1;
            else if(curChar.match(/[0-9]/))
                type = 2;
            else
                type = 3;
        }
        for (var i=selPos; i<strValue.length; i++) {
            var newtype = -1;
            var curChar = strValue.substr(i,1);
            if(curChar.match(/[A-Za-z]/))
                newtype = 1;
            else if(curChar.match(/[0-9]/))
                newtype = 2;
            if(newtype<type)
                break;
            type = newtype;
            endPos = i;
            strPinyinNow += curChar;
        }
        return [strPinyinNow,startPos,endPos];
    };
    Pinyin2Chinese.prototype.getBiaodian = function(input) {
        var element = input;
        var strValue = element.value;
        var selPos = this.doGetCaretPosition(element);
        var strPinyinNow = '';
        var type = 0;
        var startPos = selPos;
        var endPos = selPos;
        for(var i=selPos-1; i>=0; i--) {
            var newtype = -1;
            var curChar = strValue.substr(i,1);
            if(curChar.match(/[\.\,\;\!\"\'\?\:\\]/))
                newtype = 2;
            else if(curChar.match(/[0-9]/))
                newtype = 1;
            if(newtype<type)
                break;
            type = newtype;
            startPos = i;
            strPinyinNow = curChar + strPinyinNow;
        }
        type = 0;
        if(strPinyinNow.length>0) {
            endPos = selPos-1;
            var curChar = strPinyinNow.substr(strPinyinNow.length-1,1);
            if(curChar.match(/[\.\,\;\!\"\'\?\:\\]/))
                type = 1;
            else if(curChar.match(/[0-9]/))
                type = 2;
            else
                type = 3;
        }
        for(var i=selPos;i<strValue.length;i++) {
            var newtype = -1;
            var curChar = strValue.substr(i,1);
        if(curChar.match(/[\.\,\;\!\"\'\?\:\\]/))
            newtype = 1;
        else if(curChar.match(/[0-9]/))
            newtype = 2;
        if(newtype<type)
            break;
        type = newtype;
        endPos = i;
        strPinyinNow += curChar;
        }
        return [strPinyinNow,startPos,endPos];
    }
    return Pinyin2Chinese;
})();
var Tooltip = (function() {
    function Tooltip() {
        this.id = 'tt';
        this.top = 3;
        this.left = 3;
        this.maxw = 300;
        this.speed = 10;
        this.timer = 20;
        this.endalpha = 95;
        this.alpha = 0;
        this.tt = null;
        this.t = null;
        this.c = null;
        this.b = null;
        this.h = null;
        this.ie = document.all ? true : false;
    }
    Tooltip.prototype.show = function(v,w) {
        if(this.tt == null) {
            this.tt = document.createElement('div');
            this.tt.setAttribute('id', this.id);
            this.t = document.createElement('div');
            this.t.setAttribute('id', this.id + 'top');
            this.c = document.createElement('div');
            this.c.setAttribute('id', this.id + 'cont');
            this.b = document.createElement('div');
            this.b.setAttribute('id', this.id + 'bot');
            this.tt.appendChild(this.t);
            this.tt.appendChild(this.c);
            this.tt.appendChild(this.b);
            document.body.appendChild(this.tt);
            this.tt.style.opacity = 0;
            this.tt.style.filter = 'alpha(opacity=0)';
            var that = this;
            document.onmousemove = function(e){ that.pos(e) };
        }
        $(tt).css({"border-width": "1px",
            "border-color": "0x808080",
            "border-style": "solid",
            "padding": "1px",
            "position":"absolute",
            "display":"block",
            "width":"auto",
            "height": "auto",
            "background-color":"#ffffff"});
        this.c.innerHTML = v;
        this.tt.style.width = w ? w + 'px' : 'auto';
        if(!w && this.ie) {
            this.t.style.display = 'none';
            this.b.style.display = 'none';
            this.tt.style.width = this.tt.offsetWidth;
            this.t.style.display = 'block';
            this.b.style.display = 'block';
        }
        if(this.tt.offsetWidth > this.maxw) {
            this.tt.style.width = this.maxw + 'px';
        }
        this.h = parseInt(this.tt.offsetHeight) + this.top;
        clearInterval(this.tt.timer);
        var that = this;
        this.tt.timer = setInterval(function() { that.fade(1)}, this.timer);
    };

    Tooltip.prototype.pos = function(e) {
        var u = e.pageY;
        var l = e.pageX;
        this.tt.style.top = (u + 0 ) + 'px';
        this.tt.style.left = (l + this.left + 5) + 'px';
    };

    Tooltip.prototype.fade = function(d) {
        var a = this.alpha;
        if((a != this.endalpha && d == 1) || (a != 0 && d == -1)) {
            var i = this.speed;
            if(this.endalpha - a < this.speed && d == 1) {
                i = this.endalpha - a;
            } else if(this.alpha < this.speed && d == -1) {
                i = a;
            }
            this.alpha = a + (i * d);
            this.tt.style.opacity = this.alpha * .01;
            this.tt.style.filter = 'alpha(opacity=' + this.alpha + ')';
        } else {
            clearInterval(tt.timer);
            if(d == -1) {
                this.tt.style.display = 'none';
            }
        }
    };
    Tooltip.prototype.hide = function() {
        clearInterval(this.tt.timer);
        var that = this;
        this.tt.timer = setInterval(function(){that.fade(-1)}, that.timer);
    };
    return Tooltip;
})();

var pinyin2Chinese = new Pinyin2Chinese($("#Inputs")[0]);
var tooltip = new Tooltip();
$(document).keyup(function(e) {
    keycode = e.which;
    if(keycode==27) { // escape, close box, esc
        tooltip.hide();
    }
});

$(document).ready(function() {
    tooltip.show('');
    tooltip.hide();
    $("#Inputs").keyup(function(evt) {
        var keycode = evt.which;
        if(keycode==27) {
            // exc
            tooltip.hide();
        } else {
            var pinyin = pinyin2Chinese.getPinyin($(this)[0]);
            pinyin2Chinese.update(pinyin[0]);
        }
        return true;
    });
    $("#Inputs").click(function () {
        var pinyin = pinyin2Chinese.getPinyin($(this)[0]);
        pinyin2Chinese.update(pinyin[0]);
    });
    $("#Inputs").keypress( function(evt) {
        var chrTyped;
        var chrCode = evt.which;
        var keycode = evt.which;

        if(chrCode==0) {
            chrTyped = '';
        } else {
            chrTyped = String.fromCharCode(chrCode);
        }

        var pinyin = pinyin2Chinese.getPinyin($(this)[0]);

        var strHanzi = '';
        strHanzi = pinyin2Chinese.getValue();
        if(strHanzi.length>0 && (chrTyped.match(/[^0-9]/)||keycode==13)) {
            // return or no-number
            element = document.getElementById("Inputs");
            var strValue = this.value;
            var strPinyin = pinyin[0];
            selPos = pinyin2Chinese.doGetCaretPosition(element);
            strValue = strValue.substring(0, pinyin[1]) + strHanzi + strValue.substring(pinyin[2]+1, strValue.length);
            element.value = strValue;
            pinyin2Chinese.setCaretPosition(element, selPos-strPinyin.length+strHanzi.length);
            pinyin2Chinese.setValue('');
            pinyin2Chinese.update('');
            if(keycode==13) {
                // return
                if (evt.preventDefault) evt.preventDefault();
                evt.returnValue = false;
                return false;
            }
        }
        return true;
    });
});
