from flask import Blueprint, render_template, session, g
from sftm_server.auth import login_required
from sftm_server.db import get_db

bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

@bp.route('/main_dashboard')
@login_required
def main_dashboard():
    return render_template('traffic_dashboard/dashboard.html')

@bp.before_request
def save_camera():
    camera_id = session['camera_id']

    if camera_id is None:
        g.camera = None
    else:
        g.camera = get_db().execute(
            'SELECT * FROM camera WHERE c_id=?', (camera_id, )
        ).fetchone()
    