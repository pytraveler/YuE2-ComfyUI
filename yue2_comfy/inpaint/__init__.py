"""An edit of a song this pack remembers: one stretch sung again, or cut out.

``ops`` is the plan of an edit in plain Python -- which frames, how many new
ones, what becomes of the words and the score -- and runs where the tests run,
without torch. ``grid`` lays the score over the song as it was sung, so the
bars a user picks become frames, and finds where each section's singing
opens. ``core`` does what needs the model: it sings on from the edit, chooses
where the new part joins the old, draws the acoustic stage again around it
and lays the new sound into the old.
"""
