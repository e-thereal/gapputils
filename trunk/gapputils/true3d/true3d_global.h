#ifndef TRUE3D_GLOBAL_H
#define TRUE3D_GLOBAL_H

#include <QtCore/qglobal.h>

/*
Roadmap:

Stage 1:
- Qt Widget for visualizing Open GL content.
- Show screen plane and eyes
- Translation and rotation of the scene using the mouse

Stage 2:
- Build the glasses (with attached coloured balls)
- Detect balls (show Webcam image for debugging)
- Calculate ball position (show coordinates on the console)
- Visualize ball position in the 3D scene

Stage 3:
- Visualize simple 3D objects
- Adjust camera parameters according to current eye position (middle eye)
- Render the scene in 3D stereo vision

Stage 4:
- Volume rendering

*/

#ifdef TRUE3D_LIB
# define TRUE3D_EXPORT Q_DECL_EXPORT
#else
# define TRUE3D_EXPORT Q_DECL_IMPORT
#endif

#endif // TRUE3D_GLOBAL_H
