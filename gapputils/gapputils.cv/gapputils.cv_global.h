#ifndef GAPPUTILS.CV_GLOBAL_H
#define GAPPUTILS.CV_GLOBAL_H

#include <QtCore/qglobal.h>

#ifdef GAPPUTILSCVEXPORTS
# define GAPPUTILSCVAPI Q_DECL_EXPORT
#else
# define GAPPUTILSCVAPI Q_DECL_IMPORT
#endif

#endif // GAPPUTILS.CV_GLOBAL_H
