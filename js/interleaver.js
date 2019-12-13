function create_txt(x, y, txt, visible) {
    var txtLoc = new paper.Point(x, y);
    var myText = new paper.PointText(txtLoc);
    myText.fillColor = 'black';
    myText.content = txt;
    myText.fontWeight = 'bold';
    myText.fontSize = 14;
    myText.visible = visible;
    return myText;
}
var row = 3;
var col = 4;
var N = row*col;

var input = [];
create_txt(50-35, 25, 'in:', true);
for(i=0; i<N; i++) {
    txt = create_txt(50+i*38, 25, 'x['+i+']', true);
    input.push(txt);
}
var output = [];
var output_idx = [0,4,8,1,5,9,2,6,10,3,7,11];
create_txt(50-35, 25+50+60*3, 'out:', true);
for(i=0; i<N; i++) {
    var txt = create_txt(50+i*38, 25+50+60*3, 'x['+output_idx[i]+']', false);
    output.push(txt);
}
var buffer = [];
var w = 70;
var h = 60;
for(j=0; j<row;j++) {
    for(i=0;i<col;i++) {
        var rectangle = new Rectangle(new Point(50+w*i, 50+h*j), new Size(w, h));
        var path = new Path.Rectangle(rectangle);
        path.fillColor = '#e9e9ff';
        path.strokeColor = 'black'
        var txt = create_txt(50+w*i+20, 50+h*j+30, 'x['+(i+j*4)+']', false);
        buffer.push(txt);
    }
}
paper.view.viewSize.width = 50+38*N;
paper.view.viewSize.height = 100+h*row;
var n = 0;
var active = null;
function onMouseDown(event) {
    if(active) {
        active.fillColor = 'black';
        active = null;
    }
    if(n<12) {
        input[n].visible=false;
        buffer[n].visible = true;
        active = buffer[n];
    } else if(n<24) {
        output[n-12].visible=true;
        active = output[n-12];
        buffer[output_idx[n-12]].visible = false;
    }
    if(active) {
        active.fillColor = 'red';
    }
    n = n+1;
    if(n>24) {
        n=0;
        for(var i=0; i<12; i++) {
            input[i].visible=true;
            input[i].fillColor = 'black';
            output[i].visible=false;
            buffer[i].visible=false;
        }
    }
}

