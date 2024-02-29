from flask import (
    Blueprint, g, request, render_template, url_for, redirect, session)
from sftm_server.auth import login_required
from sftm_server.db import get_db

bp = Blueprint('location', __name__)

@bp.route('/location', methods=['GET','POST'])
@login_required
def location():
    if request.method == 'POST':
        selected_choice = request.form['choice']

        camera = get_db().execute(
            'SELECT * FROM camera WHERE c_name = ?', (selected_choice,)
        ).fetchone()
        
        session['camera_id'] = camera['c_id']

        return redirect(url_for('main.landing'))
    
    return render_template('location.html')
