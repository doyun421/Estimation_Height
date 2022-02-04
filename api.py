float calculatePointDistance(float coordinatesOne[], float coordinatesTwo[])
{

  float x1 = coordinatesOne[0];
  float y1 = coordinatesOne[1];
  float z1 = coordinatesOne[2];
  float x2 = coordinatesTwo[0];
  float y2 = coordinatesTwo[1];
  float z2 = coordinatesTwo[2];

  float distance = sqrt(sq(x2 - x1) + sq(y2 - y1) + sq(z2 - z1));

  return distance;
}



void sphericalToCartesian(float sphericalPoints[], float cartesianPoints[])
{

  float pan = sphericalPoints[0];
  float tilt = sphericalPoints[1];
  float distance = sphericalPoints[2];

  float x = distance * cos(degreeToRadians(pan)) * cos(degreeToRadians(tilt)); //x value
  float y = distance * cos(degreeToRadians(tilt)) * sin(degreeToRadians(pan)); //y value
  float z = distance * sin(degreeToRadians(tilt));                             //z value

  cartesianPoints[0] = x;
  cartesianPoints[1] = y;
  cartesianPoints[2] = z;
}

