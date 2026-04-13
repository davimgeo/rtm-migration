#include "geometry.hpp"

void Geometry::get()
{
};

void Geometry::load()
{
};

void Geometry::create()
{
  createReceivers();
  createSources();

  save();
}

Geometry::Receivers Geometry::createReceivers()
{
  nrec = c.nx_geom / c.offset;

  rec.x = new float[nrec];
  rec.z = new float[nrec];

  for (int i = 0; i < nrec; ++i) {
    rec.x[i] = i * c.offset;
    rec.z[i] = c.rec_depth;
  }

  return rec;
}

Geometry::Sources Geometry::createSources()
{
  src.x = new float[nsrc];
  src.z = new float[nsrc];

  for (int i = 0; i < nsrc; i++) {
    src.x 
  }
};

void Geometry::save()
{
};
