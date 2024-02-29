from flask import Blueprint, render_template
from sftm_server.auth import login_required

bp = Blueprint('main', __name__, url_prefix='/main')

@bp.route('/landing')
@login_required
def landing():
    return render_template('landing/landing.html')