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
var row = 4;
var col = 3;
var N = row*col;

var input_idx = [];
var output_idx = [];
for(i=0; i<col; i++) {
    for(j=0; j<row; j++) {
        output_idx.push(j*col+i);
    }
}
var buffer = [];
var w = 70;
var h = 60;
for(j=0; j<row; j++) {
    for(i=0; i<col; i++) {
        input_idx.push(i*row + j)
        var rect = new Rectangle(new Point(50+w*i, 50+h*j), new Size(w, h));
        var path = new Path.Rectangle(rect);
        path.fillColor = '#e9e9ff';
        path.strokeColor = 'black'
        var txt = create_txt(50+w*i+20, 50+h*j+30, 'x['+(i*row + j)+']', false);
        buffer.push(txt);
    }
}

var input = [];
create_txt(50-35, 25, 'in:', true)
for(i=0; i<N; i++) {
    var txt = create_txt(50+i*38, 25, 'x['+input_idx[i]+']', true)
    input.push(txt);
}

var output = [];
create_txt(50-35, 80+h*row, 'out:', true);
for(i=0; i<N; i++) {
    var txt = create_txt(50+i*38, 80+h*row, 'x['+i+']', false);
    output.push(txt);
}
paper.view.viewSize.width = 50+38*N;
paper.view.viewSize.height = 120+row*h;
var n = 0;
var active = null;
function onMouseDown(event) {
    if(active) {
        active.fillColor = 'black';
        active = null;
    }
    if(n<row*col) {
        input[n].visible=false;
        buffer[n].visible = true;
        active = buffer[n];
    } else if(n<N*2) {
        output[n-N].visible=true;
        active = output[n-N];
        buffer[output_idx[n-N]].visible = false;
    }
    if(active) {
        active.fillColor = 'red';
    }
    n = n+1;
    if(n>N*2) {
        n=0;
        for(var i=0; i<N; i++) {
            input[i].visible=true;
            input[i].fillColor = 'black';
            output[i].visible=false;
            buffer[i].visible=false;
        }
    }
}
