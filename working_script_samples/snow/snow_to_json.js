import fs from 'fs';

const txtFile = 'SNOW_24H_202507051300.txt';
const contents = fs.readFileSync(txtFile);

const contentsArray = contents.toString().split('\n');

let result = []
for(let line of contentsArray){
  if(!line.startsWith(20250105)){
    continue
  }
  const [createDate, stnNumber, stnName, lon, lat, stn, sd] = line.split(',').map(item => item.trim());
  // if(stn === '000-----'){
  //   continue
  // }
  const item = {
    stnNumber,
    stnName,
    lon,
    lat,
    sd
  }
  result.push(item)
}

fs.writeFileSync('SNOW_24H_202507051300_ALL.json', JSON.stringify(result));
console.log(result)