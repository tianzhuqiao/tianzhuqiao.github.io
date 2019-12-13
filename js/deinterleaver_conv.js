function create_txt(x, y, txt, visible) {
    var txtLoc = new paper.Point(x, y);
    var myText = new paper.PointText(txtLoc);
    myText.fillColor = 'black';
    myText.content = txt;
    myText.fontWeight = 'normal';
    myText.fontSize = 12;
    myText.visible = visible;
    return myText;
}
function create_pt(x, y) {
    var pt = new  Point(x, y);
    var path = new Path.Circle(pt, 2);
    path.strokeColor = 'black';
    path.fillColor = 'black';
    return pt;
}

var vectorItem = null;
function drawVector(start1, end1, start2, end2) {
    var vector = end1 - start1;
    var vector2 = end2 - start2;
    var arrowVector = vector.normalize(10);
    var arrowVector2 = vector2.normalize(10);
    if(vectorItem) {
        vectorItem.remove();
    }
	vectorItem = new Group([
		new Path([start1, end1]),
		new Path([
			end1 + arrowVector.rotate(160),
			end1,
			end1 + arrowVector.rotate(-160)
		]),
        new Path([start2, end2]),
		new Path([
			end2 + arrowVector2.rotate(160),
			end2,
			end2 + arrowVector2.rotate(-160)
		])
	]);
    vectorItem.children[1].closed = true;
    vectorItem.children[1].fillColor = '#0909ff';
    vectorItem.children[3].closed = true;
    vectorItem.children[3].fillColor = '#0909ff';
	vectorItem.strokeWidth = 0.75;
	vectorItem.strokeColor = '#0909ff';
}
var row = 5;
var w = 35;
var h = 35;

var out_all = create_txt(5, 22, 'out: ', true);
var input = create_pt(5+50, 50+50*2+18);
var in_txt = create_txt(5, 50+50*2+22, 'x[n]', true);
var output = create_pt(5+100+60*5+25+50, 50+50*2+18);
var out_txt = create_txt(15+50+60*5+25+100, 50+50*2+22, 'y[n]', true);
var bufstart = [];
var bufend = [];
var group = new Group();
var x = new Array(row);
var xi = new Array(row);
var bufs = new Array(row);
for(var i=0; i<row; i++) {
    var start = new Point(5+100, 50+50*i+18);
    var path = new Path.Circle(start, 2);
    bufstart.push(start)
    path.strokeColor = 'black';
    path.fillColor = 'black';
    var end = new Point(5+100+60*5+25, 50+50*i+18);
    var path = new Path.Circle(end, 2);
    bufend.push(end);
    path.strokeColor = 'black';
    path.fillColor = 'black';
    x[i] = new Array(row+1);
    xi[i] = new Array(row+1);
    bufs[i] = new Array(row);
    for(var j=0; j<i; j++) {
        xi[i][j]=-1;
    }
    for(var j=0; j<row-1-i; j++) {
        x[i][j]=-1;
        var cur = new Point(50+100+60*j, 50+50*i+18)
        var rectangle = new Rectangle(new Point(50+100+60*j, 50+50*i), new Size(w, h));
        var path = new Path.Rectangle(rectangle);
        path.fillColor = '#e9e9ff';
        path.strokeColor = 'black';
        bufs[i][j]=create_txt(cur.x+5, cur.y+2, ' 0 ', true);
        path = new Path(start, cur);
        start = cur;
        group.addChild(path)
    }
    group.addChild(new Path(start, end));
}
group.strokeWidth = 0.75;
group.strokeColor = '#0909ff';
paper.view.viewSize.width = 50+50*(row+3)+100;
paper.view.viewSize.height = 100+50*row;

var n=0;
function onMouseDown(event){
    var nm = n%row
    drawVector(input, bufstart[nm], bufend[nm], output);
    for(var i=nm; i>0; i--){
        xi[nm][i]= xi[nm][i-1];
    }
    xi[nm][0]=n;
    for(var i=row-nm-1; i>0; i--){
        x[nm][i]= x[nm][i-1];
    }
    x[nm][0] = xi[nm][nm];
    for(var j=0; j<row-1-nm; j++){
        if(x[nm][j]>=0) {
            bufs[nm][j].content = 'x['+x[nm][j]+']';
        }
    }
    if(x[nm][0]<0) {
        in_txt.content = '0';
    } else{
        in_txt.content = 'x['+x[nm][0]+']';
    }
    var out = '0';
    if(x[nm][row-1-nm]>=0) {
        out = 'x['+x[nm][row-1-nm]+']';
    }
    out_txt.content = 'y['+n+'] ='+out ;
    out_all.content += out + ', ';
    if(out_all.bounds.width>paper.view.viewSize.width-50) {
        out_all.content = 'out: '+out+', ';
    }
    n = n+1;
}

var btn = new Group();
var reset = create_txt(13+50, 66+50*5, 'reset', true);
reset.fontWeight = 'normal';
var path = new Path.RoundRectangle(reset.bounds.expand(10, 5), new Size(4,4));
path.fillColor = '#e9e9ff';
path.strokeColor = 'black';
btn.addChild(path);
btn.addChild(reset);
btn.onClick = function(event) {
    n = 0;
    var nm = n%row
    drawVector(input, bufstart[nm], bufend[nm], output);
    for(var i=0; i<row; i++){
        for(var j=0; j<i; j++) {
            xi[i][j] = -1;
        }
        for(var j=0; j<row-1-i; j++){
            x[i][j] = -1;
            bufs[i][j].content = '0';
        }
    }
    out_all.content = 'out: ';
    in_txt.content = 'x[0]';
    out_txt.content = 'y[0]';
}
btn.onMouseEnter = function(event) {
    path.fillColor = '#d9d9ef';
    reset.position -= new Point(1,1);
}
btn.onMouseLeave = function(event) {
    path.fillColor = '#e9e9ff';
    reset.position += new Point(1,1);
}

